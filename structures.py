# =============================================================================
# structures.py  --  Graph construction for in-situ quality prediction.
#
# A build is represented as a graph over fusing locations. This module turns
# the command data into that graph: it selects neighbours, weights the edges,
# encodes node position, and applies the causal constraint that an in-situ
# predictor faces. Everything downstream consumes what build_graph returns.
#
# NEIGHBOURS
#   Mutual k-nearest neighbours on Euclidean distance, undirected. The mutual
#   rule keeps an edge only when each location appears among the other's nk
#   nearest, so degrees fall below nk; both the nominal and the effective
#   degree are reported.
#     scope="cross"  : kNN on (x, y, z), so edges may span layers
#     scope="within" : kNN on (x, y) within each layer, giving layer-local
#                      graphs
#   Selection and ordering use geometric distance alone. Edge weights never
#   affect which locations are neighbours, so one neighbour table per
#   (scope, nk) serves every kernel mode and is worth caching.
#
# EDGE WEIGHTS
#   kernel_mode in {unweighted, spatial, spatial_temporal, full}:
#     unweighted        w = 1
#     spatial           w = ws
#     spatial_temporal  w = ws * wt
#     full              w = ws * wt * wp
#
#   ws = exp(-(d/hs)^2) with hs the commanded hatch spacing, 0.093 mm. The
#   distance is squared so the exponent is dimensionless.
#
#   wt = exp(-|d tau|/htau) on within-layer edges only, where tau is the
#   deposition phase within a layer, normalised to [0, 1].
#     - Within a layer this is the only place the information exists. Two
#       locations a few hatch lines apart can sit at opposite ends of a
#       serpentine pass, which neither the geometry nor the layer index
#       distinguishes; the median |d tau| between neighbours is about 1e-4
#       of a layer while the 99th percentile is 0.72 of one.
#     - Across layers the separation is a fixed multiple of the layer count,
#       which the causal structure of the graph already carries, and its
#       value in seconds depends on how many parts share the build plate
#       rather than on the process. Cross-layer edges take wt = 1.
#     - Normalising per layer is necessary rather than convenient: the scan
#       direction alternates by 90 degrees between layers, so a phase in one
#       layer is not comparable with the same phase in the next. It also
#       removes the need to choose a bandwidth.
#
#   Note that edge weights reach a model only through the graph Laplacian
#   from which the spectral encoding is computed. Under any other encoding
#   the kernel modes are equivalent by construction.
#
# NODE ENCODING
#   pe_mode selects between a spectral encoding (Laplacian eigenvectors), a
#   coordinate encoding (sinusoids of x, y, z and optionally tau, P, v), and
#   none. The coordinate wavelength ladder runs from each axis's extent down
#   to its process resolution -- the hatch spacing in plane, the layer
#   thickness in the build direction -- so each axis is resolved on the
#   scales at which locations actually differ. The "_dense" and "_dense12"
#   suffixes sample that same interval with 6 and 12 pairs per geometric
#   axis rather than 3.
#
#   The scales are passed in rather than measured. extents and origins come
#   from the build plan, so a location's encoding is a property of that
#   location alone and does not change when more layers are loaded. Falling
#   back to the observed minimum and range is retained only as a default for
#   standalone use.
#
#   The spectral encoding needs care to be reproducible: eigsh is given a
#   fixed random starting vector from a seeded generator, rather than the
#   all-ones vector, which lies in the null space of L. Signs follow a
#   deterministic convention and the spectral gaps are reported, since close
#   eigenvalues make the ordering unstable.
#
# CAUSAL MASK
#   A neighbour is admissible only if it lies in the current or an earlier
#   layer, the prediction timepoint being layer completion. mask_mode
#   "offline" admits every neighbour instead, which measures what the causal
#   constraint costs.
# =============================================================================

from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Physical constants of the process
# ---------------------------------------------------------------------------
HS_DEFAULT_MM = 0.093      # hatch spacing, the commanded value
T_RECOAT_DEFAULT_S = 10.0  # nominal recoat interval
LAYER_THICKNESS_MM = 0.02

# Unit of the build-time column, validated against the command data: for L31
# the column runs to 1,092,500, and 1,092,500 * 1e-6 = 1.09 s matches the
# scan duration implied by the section geometry and the commanded feed rate.
# The column is therefore a microsecond counter.
TAU_UNIT_S = 1e-6

# The temporal kernel runs on the phase normalised within each layer, so its
# bandwidth is 1 by construction. The constant exists only so that
# exp(-|dtau|/HTAU) reads as a kernel rather than a bare division.
HTAU_DEFAULT = 1.0

KERNEL_MODES = ("unweighted", "spatial", "spatial_temporal", "full")

# Coordinate modes: "noproc" drops the commanded power and speed, "notau"
# drops the deposition phase, "geom" drops both and encodes position alone.
# "_dense" and "_dense12" sample the same wavelength interval with 6 and 12
# sin/cos pairs per geometric axis rather than 3.
PE_MODES = ("spectral", "none",
            "coordinate", "coordinate_noproc", "coordinate_notau",
            "coordinate_geom",
            "coordinate_dense", "coordinate_noproc_dense",
            "coordinate_notau_dense", "coordinate_geom_dense",
            "coordinate_notau_dense12", "coordinate_geom_dense12",
            "spectral+coordinate")

SCOPES = ("cross", "within")


# ===========================================================================
# 1. Time axis  (diagnostic; the kernels use layer_phase instead)
# ===========================================================================
def build_time_axis(build_time_rows, layer_ids, layer_durations_npz,
                    t_recoat_s=T_RECOAT_DEFAULT_S, tau_unit_s=TAU_UNIT_S):
    """
    Global deposition timestamps, for reporting and for any analysis that
    needs absolute times.

    The edge weights do not consume this. Across layers the separation it
    encodes is a fixed multiple of the layer count, which the causal
    structure of the graph already carries, and its value in seconds depends
    on how many parts share the build plate rather than on the process
    itself. Within a layer, layer_phase() gives the same information on a
    scale that is comparable between layers.

    Returns
    -------
    t_full  : [N] float64 seconds -- layer offsets + within-layer tau
    t_layer : [N] float64 seconds -- layer offsets only
    """
    dat = np.load(layer_durations_npz)
    dur = dict(zip(dat["layers"].tolist(), dat["durations_s"].tolist()))

    layer_ids = np.asarray(layer_ids, dtype=int)
    used = np.unique(layer_ids)
    missing = [int(L) for L in used if int(L) not in dur]
    if missing:
        raise ValueError(f"layer_durations missing layers {missing}; "
                         f"re-run precompute_layer_durations.py")

    offset, acc = {}, 0.0
    for L in sorted(int(x) for x in used):
        offset[L] = acc
        acc += dur[L] + t_recoat_s

    off = np.array([offset[int(L)] for L in layer_ids], dtype=np.float64)
    tau = np.asarray(build_time_rows, dtype=np.float64) * tau_unit_s
    return off + tau, off.copy()


# ===========================================================================
# 2. Neighbour table (geometry only; mutual-kNN; undirected)
# ===========================================================================
def build_neighbors(xyz, layer_ids, nk, scope):
    """
    Mutual-kNN neighbour table, ordered by ascending geometric distance.

    Returns
    -------
    nbr_idx  : [N, nk] int64, -1 padding
    nbr_pad  : [N, nk] bool, True = real neighbour slot
    edges    : list of (i, j) undirected mutual edges (i < j)
    report   : degree distribution / isolated counts
    """
    assert scope in SCOPES, f"scope must be one of {SCOPES}"
    xyz = np.asarray(xyz, dtype=float)
    layer_ids = np.asarray(layer_ids, dtype=int)
    N = xyz.shape[0]

    directed = [set() for _ in range(N)]
    dist_of = {}

    def _query_block(pts, global_idx):
        n_local = len(global_idx)
        if n_local < 2:
            return
        kq = min(nk + 1, n_local)
        nn = NearestNeighbors(n_neighbors=kq, metric="euclidean")
        nn.fit(pts)
        dists, idx = nn.kneighbors(pts)
        for a in range(n_local):
            gi = int(global_idx[a])
            for b in range(kq):
                gj = int(global_idx[idx[a, b]])
                if gj == gi:
                    continue
                directed[gi].add(gj)
                dist_of[(gi, gj)] = float(dists[a, b])

    if scope == "cross":
        _query_block(xyz, np.arange(N))
    else:
        groups = defaultdict(list)
        for i, L in enumerate(layer_ids):
            groups[int(L)].append(i)
        for L, idxs in groups.items():
            idxs = np.asarray(idxs)
            _query_block(xyz[idxs, :2], idxs)

    edges = [(i, j) for i in range(N) for j in directed[i]
             if i < j and i in directed[j]]

    adj = defaultdict(list)
    for i, j in edges:
        d = dist_of.get((i, j), dist_of.get((j, i)))
        adj[i].append((d, j))
        adj[j].append((d, i))

    nbr_idx = np.full((N, nk), -1, dtype=np.int64)
    nbr_pad = np.zeros((N, nk), dtype=bool)
    for i in range(N):
        for s, (_, j) in enumerate(sorted(adj[i])[:nk]):
            nbr_idx[i, s] = j
            nbr_pad[i, s] = True

    deg = nbr_pad.sum(axis=1)
    report = {
        "scope": scope, "nk": nk, "n_nodes": N, "n_mutual_edges": len(edges),
        "deg_mean": float(deg.mean()), "deg_std": float(deg.std()),
        "deg_min": int(deg.min()), "deg_max": int(deg.max()),
        "n_isolated": int((deg == 0).sum()),
        "frac_full": float((deg == nk).mean()),
        "degree_hist": np.bincount(deg, minlength=nk + 1).tolist(),
    }
    print(f"[neighbors] scope={scope} nk={nk}  N={N:,}  "
          f"mutual_edges={len(edges):,}  deg mean/min/max="
          f"{report['deg_mean']:.1f}/{report['deg_min']}/{report['deg_max']}"
          f"  isolated={report['n_isolated']}")
    return nbr_idx, nbr_pad, edges, report


# ===========================================================================
# 3. Edge weights
# ===========================================================================
def _process_sigma_inv(P):
    """
    Pseudo-inverse of the covariance of the distinct commanded settings.

    Taking the covariance over the distinct settings rather than over their
    empirical distribution across locations keeps the kernel a measure of how
    dissimilar two settings are, independent of how many contour and infill
    points a part happens to contain.

    On a build with two joint settings the covariance has rank one, so the
    ordinary inverse does not exist and the Moore-Penrose pseudo-inverse is
    used. The kernel then takes only two values, one for a shared setting and
    one for a differing setting, and is in effect a contour/infill indicator.
    """
    uP = np.unique(P, axis=0)
    Sigma = np.cov(uP.T) if uP.shape[0] >= 2 else np.eye(P.shape[1])
    return np.linalg.pinv(Sigma)


def edge_weights(edges, xyz, P, tau_phase, layer_ids, kernel_mode,
                 hs=HS_DEFAULT_MM, htau=HTAU_DEFAULT):
    """
    Weights for the undirected edge list under the requested kernel mode.
    Returns {(i, j): w} with i < j.

    The temporal factor acts within a layer only; the module header explains
    why cross-layer edges take wt = 1 and why the phase is normalised per
    layer.
    """
    assert kernel_mode in KERNEL_MODES
    if kernel_mode == "unweighted":
        return {e: 1.0 for e in edges}

    xyz = np.asarray(xyz, dtype=float)
    P = np.asarray(P, dtype=float)
    Si = _process_sigma_inv(P)
    tau_phase = np.asarray(tau_phase, dtype=float)
    layer_ids = np.asarray(layer_ids, dtype=int)

    w = {}
    for (i, j) in edges:
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        val = np.exp(-(d / hs) ** 2)                          # ws
        if kernel_mode in ("spatial_temporal", "full"):
            if layer_ids[i] == layer_ids[j]:                  # wt
                val *= np.exp(-abs(tau_phase[i] - tau_phase[j]) / htau)
        if kernel_mode == "full":
            dp = P[i] - P[j]
            val *= np.exp(-0.5 * float(dp @ Si @ dp))         # wp
        w[(i, j)] = float(val)
    return w


# ===========================================================================
# 4. Positional encodings
# ===========================================================================
def _sign_normalize(evecs, tol=1e-8):
    """Fix the sign of each eigenvector deterministically.

    An eigenvector is defined only up to sign, so two runs of the same
    decomposition can return v and -v for the same eigenvalue. Left alone,
    that would make the encoding differ between recomputations even with a
    fixed seed. The convention here is the sign of the sum, falling back to
    the sign of the largest-magnitude entry when the sum is near zero.
    """
    out = evecs.copy()
    for c in range(out.shape[1]):
        v = out[:, c]
        s = v.sum()
        if abs(s) > tol:
            if s < 0:
                out[:, c] = -v
        else:
            if v[int(np.argmax(np.abs(v)))] < 0:
                out[:, c] = -v
    return out


def _eigsh_block(Lap, k_solve, seed=0):
    rng = np.random.RandomState(seed)
    v0 = rng.rand(Lap.shape[0])
    evals, evecs = spla.eigsh(Lap, k=k_solve, which="SM", tol=1e-6, v0=v0)
    order = np.argsort(evals)
    return evals[order], _sign_normalize(evecs[:, order])


def _adjacency(edges, weights, N):
    if not edges:
        return sp.csr_matrix((N, N), dtype=float)
    ii = [e[0] for e in edges] + [e[1] for e in edges]
    jj = [e[1] for e in edges] + [e[0] for e in edges]
    ww = [weights[e] for e in edges] * 2
    return sp.csr_matrix((ww, (ii, jj)), shape=(N, N), dtype=float)


def spectral_pe(edges, weights, layer_ids, N, k_eig, scope, seed=0):
    """
    Laplacian eigenvector encoding.
      scope="cross" : one decomposition on the whole graph; drop as many
                      trivial vectors as there are connected components.
      scope="within": per-layer decomposition; drop each layer's constant.

    Two properties are worth keeping in mind when this is compared against
    the coordinate encoding.

    It is transductive with respect to any split over layers. The
    eigenvectors of a graph are not the eigenvectors of its subgraphs, so an
    encoding computed here cannot be reproduced for a layer before the build
    reaches it.

    And it is computed on the full edge set, which includes edges to later
    layers; the causal mask is applied afterwards, to the model's
    aggregation, not to the encoding. Masking the edges here would not fix
    that, because the causal neighbourhood is directed -- a location may
    attend to those beneath it but not the reverse -- while an
    eigendecomposition requires an undirected graph. Symmetrising restores
    the edges to later layers, and a directed Laplacian gives complex
    spectra to which the construction no longer applies.
    """
    layer_ids = np.asarray(layer_ids, dtype=int)
    U = np.zeros((N, k_eig), dtype=np.float32)
    diag = {"scope": scope, "k_eig": k_eig, "blocks": []}

    if scope == "cross":
        blocks = [("all", np.arange(N))]
    else:
        groups = defaultdict(list)
        for i, L in enumerate(layer_ids):
            groups[int(L)].append(i)
        blocks = [(f"L{L}", np.asarray(v)) for L, v in sorted(groups.items())]

    edge_by_block = defaultdict(list)
    if scope == "cross":
        edge_by_block["all"] = edges
    else:
        for (i, j) in edges:
            assert layer_ids[i] == layer_ids[j], \
                "within-scope edge crosses layers?!"
            edge_by_block[f"L{layer_ids[i]}"].append((i, j))

    for name, nodes in blocks:
        n_b = len(nodes)
        if n_b < 3:
            diag["blocks"].append({"block": name, "n": n_b, "skipped": True})
            continue
        gmap = {g: l for l, g in enumerate(nodes)}
        b_edges = [(gmap[i], gmap[j]) for (i, j) in edge_by_block[name]]
        b_w = {(gmap[i], gmap[j]): weights[(i, j)]
               for (i, j) in edge_by_block[name]}
        A = _adjacency(b_edges, b_w, n_b)
        n_comp, _ = connected_components(A, directed=False)
        deg = np.asarray(A.sum(axis=1)).ravel()
        Lap = sp.diags(deg, format="csr") - A

        n_trivial = n_comp
        k_solve = min(k_eig + n_trivial, n_b - 1)
        if k_solve <= n_trivial:
            diag["blocks"].append({"block": name, "n": n_b, "skipped": True})
            continue
        evals, evecs = _eigsh_block(Lap, k_solve, seed=seed)

        nt, ev = evecs[:, n_trivial:], evals[n_trivial:]
        k_use = min(k_eig, nt.shape[1])
        U[nodes, :k_use] = nt[:, :k_use].astype(np.float32)

        nt_gaps = np.diff(evals)[n_trivial:]
        if len(nt_gaps) and len(ev) > 1:
            rel = nt_gaps / np.maximum(ev[1:len(nt_gaps) + 1], 1e-300)
            min_rel, argmax_gap = float(rel.min()), int(np.argmax(nt_gaps))
        else:
            min_rel, argmax_gap = None, None
        diag["blocks"].append({
            "block": name, "n": n_b, "n_components": int(n_comp),
            "n_isolated": int((deg == 0).sum()),
            "eval_min_nontrivial": float(ev[0]) if len(ev) else None,
            "min_gap_nontrivial": (float(nt_gaps.min())
                                   if len(nt_gaps) else None),
            "min_rel_gap_nontrivial": min_rel,
            "largest_gap_at_index": argmax_gap,
            "k_used": int(k_use),
        })
        if n_comp > 1:
            print(f"[spectral][{name}] {n_comp} components -- "
                  f"{n_comp} trivial vectors dropped")

    min_gaps = [b["min_gap_nontrivial"] for b in diag["blocks"]
                if b.get("min_gap_nontrivial") is not None]
    rel_gaps = [b["min_rel_gap_nontrivial"] for b in diag["blocks"]
                if b.get("min_rel_gap_nontrivial") is not None]
    diag["global_min_gap"] = float(min(min_gaps)) if min_gaps else None
    diag["global_min_rel_gap"] = float(min(rel_gaps)) if rel_gaps else None
    print(f"[spectral] scope={scope}  min gap = {diag['global_min_gap']}  "
          f"min RELATIVE gap = {diag['global_min_rel_gap']}")
    return U, diag


def coordinate_pe(xyz, tau_phase, P, dim, n_pairs=(3, 3, 3, 3, 1, 1),
                  extents=None, origins=None):
    """
    Sinusoidal encoding of quantities known from the build plan before
    fabrication.

    Axes, in order: x, y (in plane, mm), z (build direction, mm), tau
    (within-layer deposition phase in [0, 1]), and P and v (commanded power
    and scan speed).

    Each axis is encoded over wavelengths log-spaced from that axis's extent
    down to its process resolution -- the hatch spacing in plane, the layer
    thickness vertically. An axis therefore spends its frequencies on the
    scales at which locations along it actually differ, rather than on a
    shared generic ladder. This is what lets one construction separate
    adjacent hatch lines and adjacent layers at the same time, despite the
    two spacings differing by a factor of almost five.

    Parameters
    ----------
    xyz        : [N, 3] positions in mm
    tau_phase  : [N]    within-layer phase in [0, 1]
    P          : [N, 2] commanded (power, speed); None to omit both axes
    dim        : int    output width, zero-padded to this
    n_pairs    : sin/cos pairs per axis, in the order (x, y, z, tau, P, v)
    extents    : {axis: (coarse, fine)} wavelength range for that axis
    origins    : {axis: value} offset subtracted before encoding

    extents and origins fall back to the range and minimum observed in the
    data when not given, which makes a location's encoding depend on which
    layers happen to be loaded. Callers in main.py pass build-plan values
    instead, so that the encoding is a property of the location alone.
    """
    xyz = np.asarray(xyz, dtype=float)
    tau_phase = np.asarray(tau_phase, dtype=float)
    origins = origins or {}
    N = xyz.shape[0]

    axes = [("x", xyz[:, 0]), ("y", xyz[:, 1]), ("z", xyz[:, 2]),
            ("tau", tau_phase)]
    if P is not None:
        P = np.asarray(P, dtype=float)
        axes += [("P", P[:, 0]), ("v", P[:, 1])]

    ext = {
        "x":   (max(np.ptp(xyz[:, 0]), 1e-9), HS_DEFAULT_MM),
        "y":   (max(np.ptp(xyz[:, 1]), 1e-9), HS_DEFAULT_MM),
        "z":   (max(np.ptp(xyz[:, 2]), 1e-9), LAYER_THICKNESS_MM),
        "tau": (1.0, 1.0 / 8.0),
    }
    if P is not None:
        for name, col in (("P", P[:, 0]), ("v", P[:, 1])):
            span = max(np.ptp(col), 1e-9)
            ext[name] = (span, span)   # settings are discrete: one scale
    if extents:
        ext.update(extents)

    cols = []
    for (name, vals), npair in zip(axes, n_pairs):
        coarse, fine = ext[name]
        fine = min(fine, coarse)
        lambdas = (np.array([coarse]) if npair == 1
                   else np.geomspace(coarse, fine, npair))
        # Offset from a fixed reference rather than the observed minimum, so
        # a location's encoding does not depend on which layers are loaded.
        v0 = vals - origins.get(name, vals.min())
        for lam in lambdas:
            w = 2.0 * np.pi / max(lam, 1e-12)
            cols.append(np.sin(w * v0))
            cols.append(np.cos(w * v0))

    out = np.stack(cols, axis=1).astype(np.float32)
    if out.shape[1] < dim:
        out = np.hstack([out, np.zeros((N, dim - out.shape[1]),
                                       dtype=np.float32)])
    return out[:, :dim]


def layer_phase(build_time_rows, layer_ids, tau_unit_s=TAU_UNIT_S):
    """
    Within-layer deposition phase in [0, 1], normalised per layer.

    Layers differ in scan duration because the scan direction alternates 90
    degrees, so a rectangular section is covered by a different number of
    hatch lines on alternate layers (~1.09 s and ~0.86 s for this part).
    Per-layer normalisation makes the phase comparable in the only sense in
    which it can be: 0.5 is halfway through that layer's scan of the part,
    whatever it took and whichever direction it ran.
    """
    tau = np.asarray(build_time_rows, dtype=np.float64) * tau_unit_s
    layer_ids = np.asarray(layer_ids, dtype=int)
    phase = np.zeros_like(tau)
    for L in np.unique(layer_ids):
        m = layer_ids == L
        lo, hi = tau[m].min(), tau[m].max()
        phase[m] = (tau[m] - lo) / max(hi - lo, 1e-12)
    return phase


def _coordinate_npairs(pe_mode):
    """
    (n_pairs, uses_tau, uses_proc) for a coordinate mode.

    The suffix sets how finely the wavelength interval is sampled; the
    interval itself is unchanged. That density is selected on validation,
    like any other modelling choice.
    """
    if pe_mode.endswith("_dense12"):
        n_geom = 12
    elif pe_mode.endswith("_dense"):
        n_geom = 6
    else:
        n_geom = 3
    uses_tau = "notau" not in pe_mode and "geom" not in pe_mode
    uses_proc = "noproc" not in pe_mode and "geom" not in pe_mode
    n_pairs = [n_geom, n_geom, n_geom, 3]
    if uses_proc:
        n_pairs += [1, 1]
    return tuple(n_pairs), uses_tau, uses_proc


def build_pe(pe_mode, *, edges=None, weights=None, layer_ids=None, N=None,
             k_eig=30, scope=None, xyz=None, t_full=None, tau_phase=None,
             P=None, seed=0, extents=None, origins=None):
    """
    Entry point for every node encoding. Returns (U [N, k_eig] float32,
    diag or None).

    Modes
    -----
    none                    zeros; the control condition
    spectral                Laplacian eigenvectors of the weighted graph
    coordinate              x, y, z, tau, P, v
    coordinate_notau        without the deposition phase
    coordinate_noproc       without the commanded power and speed
    coordinate_geom         geometry alone: x, y, z
    *_dense, *_dense12      6 and 12 sin/cos pairs per geometric axis
                            rather than 3
    spectral+coordinate     both, concatenated at half the width each

    t_full is accepted so that callers need not know which modes use it, but
    is never consumed. The coordinate encoding uses the per-layer phase,
    which is comparable between layers; an absolute timestamp is not, since
    the scan direction alternates by 90 degrees.
    """
    assert pe_mode in PE_MODES, f"pe_mode must be one of {PE_MODES}"
    if pe_mode == "none":
        return np.zeros((N, k_eig), dtype=np.float32), None
    if pe_mode == "spectral":
        return spectral_pe(edges, weights, layer_ids, N, k_eig, scope,
                           seed=seed)
    if pe_mode.startswith("coordinate"):
        npairs, uses_tau, uses_proc = _coordinate_npairs(pe_mode)
        tp = tau_phase if uses_tau else np.zeros(N)
        pp = P if uses_proc else None
        width = 2 * sum(npairs)
        if width > k_eig:
            raise ValueError(
                f"pe_mode='{pe_mode}' needs {width} columns but k_eig="
                f"{k_eig}; the ladder would be silently truncated")
        return coordinate_pe(xyz, tp, pp, dim=k_eig, n_pairs=npairs,
                             extents=extents, origins=origins), None
    # spectral + coordinate, split evenly
    half = k_eig // 2
    Us, diag = spectral_pe(edges, weights, layer_ids, N, half, scope,
                           seed=seed)
    Uc = coordinate_pe(xyz, tau_phase, P, dim=k_eig - half,
                       extents=extents, origins=origins)
    return np.hstack([Us, Uc]).astype(np.float32), diag


# ===========================================================================
# 5. Causal mask
# ===========================================================================
def causal_mask(nbr_idx, nbr_pad, layer_ids, mask_mode="layer"):
    """
    Which neighbours a location is permitted to draw on.

    mask_mode="layer"  : valid iff layer(j) <= layer(i). The prediction
                         timepoint is layer completion, so a location may
                         use its own layer and every layer beneath it, and
                         nothing above.
    mask_mode="offline": every real neighbour is valid, including those in
                         later layers. Comparing the two measures what the
                         causal constraint costs.

    Padding slots are always False.
    """
    assert mask_mode in ("layer", "offline")
    layer_ids = np.asarray(layer_ids, dtype=int)
    mask = nbr_pad.copy()
    if mask_mode == "layer":
        safe = np.clip(nbr_idx, 0, None)
        later = layer_ids[safe] > layer_ids[:, None]
        mask &= ~(later & (nbr_idx >= 0))
    valid = mask.sum(axis=1)
    print(f"[mask] mode={mask_mode}  valid nbrs min/mean/max = "
          f"{valid.min()}/{valid.mean():.1f}/{valid.max()}  "
          f"fully_masked={int((valid == 0).sum())}")
    return mask


# ===========================================================================
# 6. One-call convenience wrapper
# ===========================================================================
def build_graph(df_xyz, layer_ids, P, build_time_rows, layer_durations_npz,
                *, nk, scope, kernel_mode, pe_mode, k_eig,
                hs=HS_DEFAULT_MM, htau=HTAU_DEFAULT,
                t_recoat_s=T_RECOAT_DEFAULT_S, mask_mode="layer",
                pe_seed=0, extents=None, origins=None):
    """
    Full pipeline: time axis -> neighbours -> phase -> weights -> encoding
    -> mask.

    Convenient for building one graph. For a sweep, note that the neighbour
    table depends only on (scope, nk) while the weights and the encoding do
    not, so caching the table and rebuilding only what follows is worth
    doing; main.get_graph does this.
    """
    N = len(layer_ids)
    t_full, t_layer = build_time_axis(build_time_rows, layer_ids,
                                      layer_durations_npz, t_recoat_s)
    nbr_idx, nbr_pad, edges, nbr_report = build_neighbors(
        df_xyz, layer_ids, nk, scope)
    tau_ph = layer_phase(build_time_rows, layer_ids, tau_unit_s=TAU_UNIT_S)
    w = edge_weights(edges, df_xyz, P, tau_ph, layer_ids, kernel_mode,
                     hs=hs, htau=htau)
    U, pe_diag = build_pe(pe_mode, edges=edges, weights=w,
                          layer_ids=layer_ids, N=N, k_eig=k_eig, scope=scope,
                          xyz=df_xyz, t_full=t_full, tau_phase=tau_ph, P=P,
                          seed=pe_seed, extents=extents, origins=origins)
    mask = causal_mask(nbr_idx, nbr_pad, layer_ids, mask_mode=mask_mode)
    return {
        "t_full": t_full, "t_layer": t_layer,
        "nbr_idx": nbr_idx, "nbr_pad": nbr_pad, "nbr_mask": mask,
        "edges": edges, "edge_weights": w,
        "U": U, "pe_diag": pe_diag, "nbr_report": nbr_report,
        "config": {"nk": nk, "scope": scope, "kernel_mode": kernel_mode,
                   "pe_mode": pe_mode, "k_eig": k_eig, "hs": hs,
                   "htau": htau, "t_recoat_s": t_recoat_s,
                   "mask_mode": mask_mode, "pe_seed": pe_seed,
                   "sign_rule": "sum_then_argmax_v1"},
    }
