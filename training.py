# =============================================================================
# training.py  --  Shared training machinery (merged from the two legacy
# mains, which were byte-identical here) + revision protocol additions:
#
#   * seed_everything(): one seed drives torch / numpy / random / cuda /
#     DataLoader generator & workers.
#   * run_seeds(): the 5-seed protocol. Sweep phase uses ONE fixed seed on
#     validation; the selected config is re-trained on SEED_SET and test
#     metrics reported as mean±std. Per-seed predictions are saved so
#     paired tests (same seeds, same test set) are possible downstream.
#   * All rows logged with seed, config string and git commit (if any) —
#     the (seed, config, commit) provenance triple for R1 #14.
#   * profiling.RunProfiler wired in: train time, inference latency,
#     peak GPU memory land in results/compute_summary.csv.
# =============================================================================

import copy
import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from profiling import RunProfiler

# ---------------------------------------------------------------------------
SEED_SET   = [0, 1, 2, 3, 4]     # shared by ALL models (paired comparisons)
NUM_WORKERS = int(os.environ.get("STGT_NUM_WORKERS", "0"))
SWEEP_SEED = 0                   # single seed for hyperparameter sweeps
EPOCHS     = 100
PATIENCE   = 10
MIN_EPOCHS = 25

FORCE_RERUN = False      # set True to ignore completed runs and retrain
SAVE_CKPT = False        # save best model weights to results/ckpt/

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, seed, shuffle=False, num_workers=0):
    g = torch.Generator().manual_seed(seed) if shuffle else None

    def _worker_init(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    kw = {}
    if num_workers:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=g, num_workers=num_workers,
                      worker_init_fn=_worker_init if num_workers else None,
                      pin_memory=(device == "cuda"), **kw)


# ---------------------------------------------------------------------------
# Metrics / epochs (unchanged logic from legacy, single copy)
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {"mse": float(mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2":  float(r2_score(y_true, y_pred))}


def train_one_epoch(model, loader, optimizer, scaler, criterion, forward_fn):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device, enabled=use_amp):
            _, loss = forward_fn(*batch, criterion=criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        b = batch[0].size(0)
        tot += loss.detach().item() * b
        n += b
    return tot / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, forward_fn):
    model.eval()
    tot, n = 0.0, 0
    yt, yp = [], []
    for batch in loader:
        with autocast(device_type=device, enabled=use_amp):
            pred, loss = forward_fn(*batch, criterion=criterion)
        b = batch[0].size(0)
        tot += loss.detach().item() * b
        n += b
        yt.append(batch[1].cpu().numpy())
        yp.append(pred.detach().float().cpu().numpy())
    return tot / max(n, 1), np.concatenate(yt).ravel(), \
        np.concatenate(yp).ravel()


def get_optimizer_config(model_type):
    m = model_type.lower()
    if m in ("cnn", "resnet", "cnn_lstm"):
        return {"lr": 1e-3, "weight_decay": 1e-4}
    if m in ("vit", "video_vit", "stgt", "gat", "gnn"):
        return {"lr": 5e-4, "weight_decay": 1e-3}
    raise ValueError(f"Unknown model_type: {model_type}")


def run_training(model, tr_loader, va_loader, forward_fn, tag, model_type,
                 epochs=EPOCHS, verbose=True):
    """Cosine LR, warmup-before-patience early stopping (legacy logic)."""
    criterion = nn.MSELoss()
    cfg = get_optimizer_config(model_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp)

    best_val, wait, best_epoch = float("inf"), 0, -1
    best_state = None
    history = {"tr_mse": [], "va_mse": [], "lr": []}

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, tr_loader, optimizer, scaler,
                             criterion, forward_fn)
        va, _, _ = eval_one_epoch(model, va_loader, criterion, forward_fn)
        scheduler.step()
        history["tr_mse"].append(tr)
        history["va_mse"].append(va)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if va < best_val - 1e-8:
            best_val, best_epoch, wait = va, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        elif ep >= MIN_EPOCHS:
            wait += 1
            if wait >= PATIENCE:
                if verbose:
                    print(f"  [{tag}] early stop ep {ep} "
                          f"best={best_val:.6f}@{best_epoch}")
                break
        if verbose:
            print(f"  [{tag}] ep {ep:03d} tr={tr:.6f} va={va:.6f} "
                  f"best={best_val:.6f} wait={wait}/{PATIENCE}")

    if best_state is None:                      # safety guard (kept)
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs
    return best_state, best_epoch, history


# ---------------------------------------------------------------------------
# The 5-seed protocol
# ---------------------------------------------------------------------------
def _row_path(results_dir, tag, seed, subdir="rows"):
    return os.path.join(results_dir, subdir, f"{tag}__seed{seed}.json")


def _write_row(results_dir, tag, seed, row, subdir="rows"):
    """
    Write one run's metrics to its OWN file, atomically.

    Each (tag, seed) is produced by exactly one process, so there is no
    write contention even when several Slurm jobs run concurrently. The
    consolidated summary CSVs are rebuilt from these files by
    merge_results.py (or automatically at the end of run_seeds).
    """
    d = os.path.join(results_dir, subdir)
    os.makedirs(d, exist_ok=True)
    path = _row_path(results_dir, tag, seed, subdir)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, path)          # atomic on POSIX


def collect_rows(results_dir="results", subdir="rows"):
    """All per-run rows in a subdirectory as a DataFrame (empty if none)."""
    d = os.path.join(results_dir, subdir)
    if not os.path.isdir(d):
        return pd.DataFrame()
    recs = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn)) as f:
                recs.append(json.load(f))
        except Exception as e:
            print(f"[collect] skipping unreadable {fn}: {e}")
    return pd.DataFrame(recs)


def rebuild_summaries(results_dir="results"):
    """
    Regenerate summary_seeds.csv and summary_agg.csv from the per-run row
    files. Safe to call at any time and from any job: the row files are the
    source of truth, the CSVs are just a convenient view.
    """
    for subdir, out in (("rows", "summary_seeds.csv"),
                        ("agg", "summary_agg.csv")):
        df = collect_rows(results_dir, subdir)
        if df.empty:
            continue
        sort_cols = [c for c in ("model", "seed") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        tmp = os.path.join(results_dir, out + f".tmp.{os.getpid()}")
        df.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(results_dir, out))
    return True


def _completed_row(tag, seed, results_dir, need_ckpt=False):
    if FORCE_RERUN:
        return None
    preds = f"{results_dir}/{tag}_seed{seed}_preds_test.csv"
    if not os.path.exists(preds):
        return None
    if need_ckpt and not os.path.exists(
            f"{results_dir}/ckpt/{tag}_seed{seed}.pt"):
        return None

    row = _row_path(results_dir, tag, seed)
    if os.path.exists(row):
        try:
            with open(row) as f:
                return json.load(f)
        except Exception:
            pass

    # legacy fallback: rows written into the shared CSV
    legacy = f"{results_dir}/summary_seeds.csv"
    if os.path.exists(legacy):
        try:
            df = pd.read_csv(legacy)
            m = (df["model"] == tag) & (df["seed"] == seed)
            if m.any():
                rec = df[m].iloc[-1].to_dict()
                _write_row(results_dir, tag, seed, rec)   # migrate forward
                print(f"  [{tag}_seed{seed}] migrated legacy summary row")
                return rec
        except Exception:
            pass
    return None


def run_seeds(build_model_fn, build_forward_fn, datasets, tag, model_type,
              batch_size, config_str="", seeds=SEED_SET, results_dir="results",
              save_ckpt=None,
              n_test_layers=None, graph_precompute_s=None,
              eig_precompute_s=None):
    """
    Train the SAME configuration under each seed; report mean±std on test.

    Seeds whose outputs are already on disk are skipped and their logged
    metrics reused, so an interrupted session can simply be restarted.
    Set FORCE_RERUN = True to retrain regardless.

    build_model_fn(seed)            -> fresh model on device
    build_forward_fn(model)         -> forward_fn(*batch, criterion=)
    datasets = (tr_ds, va_ds, te_ds)
    config_str : free-form provenance string (kernel_mode, nk, k, hs, ht...)

    Writes:
      results/{tag}_seed{S}_preds_test.csv   (per-seed, for paired tests)
      results/{tag}_seed{S}_history.csv      (per-epoch record)
      results/summary_seeds.csv              (one row per seed)
      results/summary_agg.csv                (mean±std row per tag)
      results/compute_summary.csv            (profiler, first trained seed)
    """
    os.makedirs(results_dir, exist_ok=True)
    save_ckpt = SAVE_CKPT if save_ckpt is None else save_ckpt
    tr_ds, va_ds, te_ds = datasets
    commit = git_commit()
    per_seed = []
    n_skipped = 0

    for si, seed in enumerate(seeds):
        done = _completed_row(tag, seed, results_dir, need_ckpt=save_ckpt)
        if done is not None:
            per_seed.append(done)
            n_skipped += 1
            print(f"  [{tag}_seed{seed}] already complete "
                  f"(test R2={done.get('test_r2', float('nan')):.4f}) — skipped")
            continue

        seed_everything(seed)
        tr_loader = make_loader(tr_ds, batch_size, seed, shuffle=True,
                                num_workers=NUM_WORKERS)
        va_loader = make_loader(va_ds, batch_size, seed,
                                num_workers=NUM_WORKERS)
        te_loader = make_loader(te_ds, batch_size, seed,
                                num_workers=NUM_WORKERS)

        model = build_model_fn(seed)
        fwd = build_forward_fn(model)
        stag = f"{tag}_seed{seed}"
        first_trained = (len(per_seed) == n_skipped)   # nothing trained yet

        prof = RunProfiler(stag, device) if first_trained else None
        if prof:
            prof.log_params(model)
            prof.log_precompute(graph_s=graph_precompute_s,
                                eig_s=eig_precompute_s)
            prof.start_train()

        best_state, best_ep, hist = run_training(
            model, tr_loader, va_loader, fwd, stag, model_type,
            verbose=first_trained)
        model.load_state_dict(best_state)
        if save_ckpt:
            ckpt_dir = os.path.join(results_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model_state": best_state, "tag": tag, "seed": seed,
                        "best_epoch": best_ep, "config": config_str,
                        "commit": commit},
                       os.path.join(ckpt_dir, f"{stag}.pt"))
        save_history(hist, best_ep, stag, results_dir, plot=first_trained)

        if prof:
            prof.end_train(len(hist["tr_mse"]), best_epoch=best_ep)
            prof.measure_inference(model, fwd, te_loader,
                                   n_test_layers=n_test_layers)
            prof.extra(config=config_str, commit=commit)
            prof.write(f"{results_dir}/compute_summary.csv")

        criterion = nn.MSELoss()
        row = {"model": tag, "seed": seed, "best_epoch": best_ep,
               "epochs_run": len(hist["tr_mse"]),
               "config": config_str, "commit": commit}
        for split, loader in [("train", tr_loader), ("val", va_loader),
                              ("test", te_loader)]:
            _, yt, yp = eval_one_epoch(model, loader, criterion, fwd)
            m = compute_metrics(yt, yp)
            row.update({f"{split}_{k}": v for k, v in m.items()})
            if split == "test":
                pd.DataFrame({"y_true": yt, "y_pred": yp}).to_csv(
                    f"{results_dir}/{stag}_preds_test.csv", index=False)
        per_seed.append(row)
        print(f"  [{stag}] test R2={row['test_r2']:.4f}")

        _write_row(results_dir, tag, seed, row)

    # aggregate
    df = pd.DataFrame(per_seed)
    agg = {"model": tag, "config": config_str, "commit": commit,
           "n_seeds": len(seeds)}
    for c in [c for c in df.columns if c.startswith(("train_", "val_",
                                                     "test_"))]:
        agg[f"{c}_mean"] = float(df[c].mean())
        agg[f"{c}_std"] = float(df[c].std(ddof=1))
    _write_row(results_dir, tag, "agg", agg, subdir="agg")
    rebuild_summaries(results_dir)
    note = f"  ({n_skipped}/{len(seeds)} seeds reused from disk)" \
        if n_skipped else ""
    print(f"[{tag}] test R2 = {agg['test_r2_mean']:.4f} "
          f"± {agg['test_r2_std']:.4f}  (n={len(seeds)}){note}")
    return agg


def save_history(history, best_epoch, tag, results_dir="results",
                 plot=True):
    """
    Persist the per-epoch training record.
      results/{tag}_history.csv  — epoch, tr_mse, va_mse, lr (every run)
      results/{tag}_curve.png    — curve with best epoch + warmup marked
                                   (only when plot=True, i.e. seed 0)
    """
    os.makedirs(results_dir, exist_ok=True)
    n = len(history["tr_mse"])
    pd.DataFrame({
        "epoch": np.arange(1, n + 1),
        "tr_mse": history["tr_mse"],
        "va_mse": history["va_mse"],
        "lr": history["lr"],
    }).to_csv(f"{results_dir}/{tag}_history.csv", index=False)

    if not plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eps = np.arange(1, n + 1)
        plt.figure(figsize=(7, 4.5))
        plt.plot(eps, history["tr_mse"], label="Train MSE")
        plt.plot(eps, history["va_mse"], label="Val MSE")
        plt.axvline(best_epoch, ls="--", c="k",
                    label=f"Best epoch {best_epoch}")
        if n > MIN_EPOCHS:
            plt.axvline(MIN_EPOCHS, ls=":", c="gray",
                        label=f"Warmup ends ({MIN_EPOCHS})")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("MSE (normalised, log scale)")
        plt.title(f"Training curve — {tag}  ({n} epochs run)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{tag}_curve.png", dpi=150)
        plt.close()
    except Exception as e:                      # never kill a run over a plot
        print(f"[history] plot failed for {tag}: {e}")


def _append(path, row):
    df_new = pd.DataFrame([row])
    if os.path.exists(path):
        old = pd.read_csv(path)
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def paired_comparison(results_dir, tag_a, tag_b, seeds=SEED_SET):
    """
    Seed-paired comparison of per-sample squared errors -> per-seed R2 diff,
    plus a paired t-test and Wilcoxon on the per-seed test R2 values.
    """
    from scipy import stats
    r2a, r2b = [], []
    for s in seeds:
        pa = pd.read_csv(f"{results_dir}/{tag_a}_seed{s}_preds_test.csv")
        pb = pd.read_csv(f"{results_dir}/{tag_b}_seed{s}_preds_test.csv")
        r2a.append(r2_score(pa.y_true, pa.y_pred))
        r2b.append(r2_score(pb.y_true, pb.y_pred))
    d = np.array(r2a) - np.array(r2b)
    t = stats.ttest_rel(r2a, r2b)
    try:
        wsr = stats.wilcoxon(r2a, r2b)
        w_p = float(wsr.pvalue)
    except ValueError:
        w_p = float("nan")
    out = {"a": tag_a, "b": tag_b,
           "r2_a_mean": float(np.mean(r2a)), "r2_b_mean": float(np.mean(r2b)),
           "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
           "paired_t_p": float(t.pvalue), "wilcoxon_p": w_p}
    print(f"[paired] {tag_a} vs {tag_b}: dR2={out['diff_mean']:+.4f} "
          f"p_t={out['paired_t_p']:.4f} p_w={out['wilcoxon_p']:.4f}")
    return out
