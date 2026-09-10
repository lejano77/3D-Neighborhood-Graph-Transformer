# =============================================================================
# data_preprocessing.py  --  Data loading and unified index construction.
#
# The three data sources are aligned here and nowhere else. Each fusing
# location has a row in the command CSV (coordinates, commanded parameters,
# melt-pool scalars, XCT value) and a frame in the layer's TIFF stack. The
# functions below build roi_all, row_index and ys in a single loop under one
# filter set, so their lengths and orderings are consistent by construction
# rather than by convention. The asserts check that they stay that way.
#
# Images are loaded on demand through load_image() rather than held in
# memory: a graph model reads about twenty images per sample, so a full
# tensor would be both large and mostly unused within any one batch.
#
# All paths are function arguments; nothing here is hardcoded to a machine.
# =============================================================================
import numpy as np
import pandas as pd
import tifffile as tiff

# ---------------------------------------------------------------------------
# Column schema of the 40-column part1/LXXXX.csv
# ---------------------------------------------------------------------------
CSV_NAMES = [
    "Part number", "Build time",
    "Command laser position in X", "Command laser position in Y",
    "Command laser power", "Command scan speed",
    "Real laser position in X", "Real laser position in Y",
    "Real laser power", "Real scan speed",
    "Melt pool length 1", "Melt pool width 1", "Melt pool area 1",
    "Melt pool length 2", "Melt pool width 2", "Melt pool area 2",
    "Melt pool length 3", "Melt pool width 3", "Melt pool area 3",
] + [f"LWI pixel value {i}" for i in range(18)] \
  + ["XCT voxel value 1", "XCT voxel value 2", "XCT voxel value 3"]

MP_COLS = ["Melt pool length 1", "Melt pool width 1", "Melt pool area 1"]

# Target variable: column 38, the unfiltered XCT voxel value. Columns 39 and
# 40 hold 3x3x3 and 5x5x5 mean-filtered versions. With 12 um isotropic voxels
# against a 20 um layer thickness, a 3x3x3 window spans about 1.8 layers, so
# a filtered target would be partly defined by the neighbouring layers whose
# contribution this work sets out to measure. The unfiltered column keeps the
# target independent of the graph.
XCT_COL = "XCT voxel value 1"

TCOL    = "Build time"      # per-layer scan clock, in microseconds
XCOL    = "Command laser position in X"
YCOL    = "Command laser position in Y"
ZCOL    = "Command laser position in Z"
PCOL    = "Command laser power"
VCOL    = "Command scan speed"

LAYER_THICKNESS_MM = 0.02


def load_layer_tables(layers, part_dir="./part1"):
    """Read part1/LXXXX.csv for each layer; frame_idx = original row index."""
    tables = {}
    for L in layers:
        df = pd.read_csv(f"{part_dir}/L{L:04d}.csv", header=None)
        df.columns = CSV_NAMES
        df = df.reset_index().rename(columns={"index": "frame_idx"})
        df[ZCOL] = L * LAYER_THICKNESS_MM
        tables[L] = df
    return tables


def load_stacks(layers, tif_dir, tif_pattern="MPMcamera_L{:04d}.tif"):
    """Load MPM TIF stacks per layer as numpy arrays [n, H, W]."""
    stacks = {}
    for L in layers:
        stack = tiff.imread(f"{tif_dir}/" + tif_pattern.format(L))
        if stack.ndim == 2:
            stack = stack[None, ...]
        stacks[L] = stack
    return stacks


def build_unified_index(layers, part_dir, tif_dir,
                        tif_pattern="MPMcamera_L{:04d}.tif"):
    """
    THE single data path. One loop, one filter set:
      keep row iff  XCT finite  AND  all melt-pool metrics nonzero
                    AND  frame_idx < TIF stack size (frame validity guard).

    Returns
    -------
    roi_all   : DataFrame [N] with frame_idx, Build time, X/Y/Z, P/V,
                melt-pool cols, XCT, layer, layer_id
    row_index : list of (L, frame_idx), len N — image lookup key
    y_np      : [N, 1] float32 raw XCT target
    stacks    : dict L -> numpy stack (for load_image)
    """
    tables = load_layer_tables(layers, part_dir)
    stacks = load_stacks(layers, tif_dir, tif_pattern)

    rows, row_index, ys = [], [], []
    n_drop = {"xct": 0, "mp0": 0, "frame": 0}

    for L in layers:
        t = tables[L].copy()
        t[XCT_COL] = pd.to_numeric(t[XCT_COL], errors="coerce")
        t = t.sort_values("frame_idx").reset_index(drop=True)
        stack_size = stacks[L].shape[0]

        for pos in range(len(t)):
            yv = t[XCT_COL].iloc[pos]
            fri = int(t["frame_idx"].iloc[pos])
            if not np.isfinite(yv):
                n_drop["xct"] += 1
                continue
            if (t[MP_COLS].iloc[pos] == 0).any():
                n_drop["mp0"] += 1
                continue
            if fri >= stack_size:
                n_drop["frame"] += 1
                continue
            rows.append({
                "frame_idx": fri,
                TCOL:  float(t[TCOL].iloc[pos]),
                XCOL:  float(t[XCOL].iloc[pos]),
                YCOL:  float(t[YCOL].iloc[pos]),
                ZCOL:  float(t[ZCOL].iloc[pos]),
                PCOL:  float(t[PCOL].iloc[pos]),
                VCOL:  float(t[VCOL].iloc[pos]),
                **{c: float(t[c].iloc[pos]) for c in MP_COLS},
                XCT_COL: float(yv),
                "layer": L, "layer_id": L,
            })
            row_index.append((L, fri))
            ys.append(float(yv))

    roi_all = pd.DataFrame(rows).reset_index(drop=True)
    y_np = np.asarray(ys, dtype=np.float32).reshape(-1, 1)

    assert len(roi_all) == len(row_index) == len(y_np), \
        "unified index broke — this must never happen"
    # frame validity double-check
    for L, fri in row_index:
        assert fri < stacks[L].shape[0]

    print(f"[index] {len(roi_all):,} valid samples over {len(layers)} layers "
          f"(dropped: xct={n_drop['xct']}, mp0={n_drop['mp0']}, "
          f"frame={n_drop['frame']})")
    return roi_all, row_index, y_np, stacks


def make_image_loader(stacks, image_size=120):
    """Returns load_image(L, frame_idx) -> float32 numpy [1, H, W] in [0,1]."""
    def load_image(L, frame_idx):
        stack = stacks[L]
        if frame_idx >= stack.shape[0]:
            return np.zeros((1, image_size, image_size), dtype=np.float32)
        return stack[frame_idx].astype(np.float32)[None, ...] / 255.0
    return load_image


def graph_inputs(roi_all):
    """Arrays structures.build_graph expects, in roi_all order."""
    return {
        "df_xyz": roi_all[[XCOL, YCOL, ZCOL]].to_numpy(float),
        "layer_ids": roi_all["layer_id"].to_numpy(int),
        "P": roi_all[[PCOL, VCOL]].to_numpy(float),
        "build_time_rows": roi_all[TCOL].to_numpy(float),
    }
