# =============================================================================
# profiling.py  --  Compute-cost instrumentation.
#
# Whether a model can run in situ depends on how its inference latency
# compares with the layer cycle of the process, so cost is measured
# alongside accuracy rather than estimated afterwards.
#
# Produces per-model rows for results/compute_summary.csv:
#   n_params, train_time_s, epochs_run, s_per_epoch,
#   infer_ms_per_sample, infer_samples_per_s, infer_s_per_layer,
#   peak_gpu_mem_mb (cuda only), graph_precompute_s, eig_precompute_s
#
# infer_s_per_layer is the figure to compare against the layer cycle: it
# scales the per-sample latency by the number of locations in a layer.
#
# Usage, from the training loop:
#   prof = RunProfiler(tag, device)
#   prof.start_train(); ... epochs ...; prof.end_train(n_epochs)
#   prof.measure_inference(model, forward_fn, te_loader, n_test_layers=4)
#   prof.log_precompute(graph_s=..., eig_s=...)
#   prof.write("results/compute_summary.csv")
# =============================================================================

import csv
import os
import time
from contextlib import contextmanager

import torch


@contextmanager
def timer():
    """with timer() as t: ...; elapsed = t()  (seconds)."""
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class RunProfiler:
    def __init__(self, tag, device):
        self.tag = tag
        self.device = str(device)
        self.row = {"model": tag, "device": self.device}
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # ----- training ---------------------------------------------------------
    def start_train(self):
        _sync(self.device)
        self._t0 = time.perf_counter()

    def end_train(self, n_epochs, best_epoch=None):
        _sync(self.device)
        dt = time.perf_counter() - self._t0
        self.row.update({
            "train_time_s": round(dt, 2),
            "epochs_run": int(n_epochs),
            "s_per_epoch": round(dt / max(n_epochs, 1), 3),
        })
        if best_epoch is not None:
            self.row["best_epoch"] = int(best_epoch)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.row["peak_gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated() / 2**20, 1)

    # ----- inference --------------------------------------------------------
    @torch.no_grad()
    def measure_inference(self, model, forward_fn, loader,
                          n_test_layers=None, n_warmup_batches=3):
        """
        Timed forward passes over `loader` (batch pipeline included — the
        deployment-relevant number, since neighbor image gathering is part
        of the cost). Reports per-sample latency and per-layer throughput.
        """
        model.eval()
        it = iter(loader)
        for _ in range(n_warmup_batches):          # warmup (cudnn autotune)
            try:
                batch = next(it)
            except StopIteration:
                break
            forward_fn(*batch, criterion=None)
        _sync(self.device)

        n, t0 = 0, time.perf_counter()
        for batch in loader:
            forward_fn(*batch, criterion=None)
            n += batch[0].size(0)
        _sync(self.device)
        dt = time.perf_counter() - t0

        ms = 1e3 * dt / max(n, 1)
        self.row.update({
            "infer_n_samples": n,
            "infer_ms_per_sample": round(ms, 3),
            "infer_samples_per_s": round(n / dt, 1) if dt > 0 else None,
        })
        if n_test_layers:
            self.row["infer_s_per_layer"] = round(dt / n_test_layers, 3)

    # ----- one-off precompute costs (graph build, eigendecomposition) ------
    def log_precompute(self, graph_s=None, eig_s=None):
        if graph_s is not None:
            self.row["graph_precompute_s"] = round(graph_s, 2)
        if eig_s is not None:
            self.row["eig_precompute_s"] = round(eig_s, 2)

    def log_params(self, model):
        self.row["n_params"] = count_params(model)

    def extra(self, **kv):
        self.row.update(kv)

    # ----- persist ----------------------------------------------------------
    def write(self, path="results/compute_summary.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        # union of columns with what's already on disk
        fieldnames = list(self.row.keys())
        rows = []
        if exists:
            with open(path, newline="") as f:
                r = csv.DictReader(f)
                old = r.fieldnames or []
                rows = list(r)
            fieldnames = old + [c for c in self.row if c not in old]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for old_row in rows:
                w.writerow(old_row)
            w.writerow(self.row)
        print(f"[profile] {self.tag}: {self.row}")
