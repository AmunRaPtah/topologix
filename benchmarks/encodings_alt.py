"""Four topological encodings of a ligand point cloud, varying three axes.

V  Vietoris-Rips + homology + persistence image   (control, replicates Experiment 2)
X  alpha complex + homology + persistence image   (varies the filtration)
L  Vietoris-Rips + homology + persistence landscape (varies the vectorization)
S  epsilon-graph sweep + persistent Laplacian spectrum (varies the algebraic object)

All four use the same seven element channels as Experiment 2, so the comparison against the
ESPH result is like-for-like.
"""

import gudhi
import numpy as np
from persim import PersistenceImager
from ripser import ripser
from scipy.linalg import eigvalsh
from scipy.spatial.distance import pdist, squareform

THRESH = 8.0
CHANNELS = ["C", "N", "O", ("C", "N"), ("C", "O"), ("N", "O"), "ALL"]

_IMGR = PersistenceImager(birth_range=(0.0, 8.0), pers_range=(0.0, 8.0), pixel_size=2.0)
_IMG_LEN = 16
_H0_BINS = 8

LAND_RES = 10          # landscape samples per level
LAND_LEVELS = 3        # number of landscape levels
EPS_SWEEP = np.linspace(1.0, 8.0, 16)
LAP_STATS = 6


def _select(P, elems, which):
    if which == "ALL":
        return P
    keep = np.isin(elems, list(which) if isinstance(which, tuple) else [which])
    return P[keep]


def _summary(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(4)
    p = dgm[:, 1] - dgm[:, 0]
    return np.array([len(p), p.max(), p.sum(), p.mean()])


def _clean(dgm):
    if dgm is None or dgm.size == 0:
        return np.empty((0, 2))
    d = np.asarray(dgm, float)
    return d[np.isfinite(d).all(1)]


def _h0hist(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(_H0_BINS)
    h, _ = np.histogram(np.clip(dgm[:, 1], 0, THRESH), bins=_H0_BINS, range=(0, THRESH))
    return h.astype(float)


def _image(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(_IMG_LEN)
    return np.asarray(_IMGR.transform(dgm, skew=True), float).ravel()


def _landscape(dgm):
    """Persistence landscape sampled on a fixed grid. Vectorization axis."""
    out = np.zeros(LAND_LEVELS * LAND_RES)
    if dgm.shape[0] == 0:
        return out
    grid = np.linspace(0.0, THRESH, LAND_RES)
    b, d = dgm[:, 0], dgm[:, 1]
    # tent function of each bar evaluated on the grid, then k-th largest per grid point
    tents = np.minimum(grid[None, :] - b[:, None], d[:, None] - grid[None, :])
    tents = np.maximum(tents, 0.0)
    tents = np.sort(tents, axis=0)[::-1]
    for k in range(min(LAND_LEVELS, tents.shape[0])):
        out[k * LAND_RES:(k + 1) * LAND_RES] = tents[k]
    return out


def _vr_diagrams(P):
    # Pass an explicit distance matrix. A raw (n,3) point cloud with n == 3 is square, and
    # ripser then cannot tell a point cloud from a distance matrix; element channels with
    # exactly three atoms hit that case. The prior ESPH code passed distance matrices
    # explicitly and is unaffected.
    D = squareform(pdist(P))
    dg = ripser(D, distance_matrix=True, maxdim=1, thresh=THRESH)["dgms"]
    return _clean(dg[0]), _clean(dg[1] if len(dg) > 1 else None)


def encode_V(P):
    d0, d1 = _vr_diagrams(P)
    return np.concatenate([_h0hist(d0), _summary(d0), _image(d1), _summary(d1)])


def encode_X(P):
    """Alpha complex. Filtration values are squared circumradii; take sqrt for comparability."""
    if len(P) < 4:
        return np.zeros(_H0_BINS + 4 + _IMG_LEN + 4)
    st = gudhi.AlphaComplex(points=P.tolist()).create_simplex_tree()
    st.compute_persistence()
    d0 = _clean(st.persistence_intervals_in_dimension(0))
    d1 = _clean(st.persistence_intervals_in_dimension(1))
    for d in (d0, d1):
        if d.size:
            np.sqrt(d, out=d)
    d0 = d0[d0[:, 1] <= THRESH] if d0.size else d0
    d1 = d1[d1[:, 1] <= THRESH] if d1.size else d1
    return np.concatenate([_h0hist(d0), _summary(d0), _image(d1), _summary(d1)])


def encode_L(P):
    d0, d1 = _vr_diagrams(P)
    return np.concatenate([_landscape(d0), _landscape(d1)])


def encode_S(P):
    """Dimension-0 persistent Laplacian spectrum across an epsilon sweep.

    L0 = D - A on the epsilon-neighbourhood graph. Its zero-eigenvalue multiplicity IS the
    Betti-0 number, so homology is nested inside this block; the non-zero spectrum is the
    non-harmonic information that homology discards.
    """
    out = np.zeros(len(EPS_SWEEP) * LAP_STATS)
    n = len(P)
    if n < 2:
        return out
    D = squareform(pdist(P))
    for i, eps in enumerate(EPS_SWEEP):
        A = (D <= eps).astype(float)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        w = np.clip(eigvalsh(L), 0.0, None)
        tol = 1e-8 * max(1.0, w.max())
        nz = w[w > tol]
        beta0 = float((w <= tol).sum())
        out[i * LAP_STATS:(i + 1) * LAP_STATS] = [
            beta0,
            nz.min() if nz.size else 0.0,      # Fiedler value
            w.max(),
            nz.mean() if nz.size else 0.0,
            w.sum(),
            (w.max() - (nz.min() if nz.size else 0.0)),   # spectral gap
        ]
    return out


ENCODERS = {"V": encode_V, "X": encode_X, "L": encode_L, "S": encode_S}
LENGTHS = {"V": _H0_BINS + 4 + _IMG_LEN + 4,
           "X": _H0_BINS + 4 + _IMG_LEN + 4,
           "L": 2 * LAND_LEVELS * LAND_RES,
           "S": len(EPS_SWEEP) * LAP_STATS}


def encode_all_channels(P, elems):
    """Returns {block: concatenated-over-channels vector}."""
    out = {k: [] for k in ENCODERS}
    for ch in CHANNELS:
        Q = _select(P, elems, ch)
        for k, fn in ENCODERS.items():
            out[k].append(fn(Q) if len(Q) >= 2 else np.zeros(LENGTHS[k]))
    return {k: np.concatenate(v) for k, v in out.items()}
