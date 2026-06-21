"""Filtration + persistent homology -> fixed-length feature vectors.

Vietoris-Rips PH on a 3D point cloud (ripser), vectorized to a reproducible,
fixed-length descriptor so molecules of different atom counts map to the same
feature space. Prior art: Cang & Wei (element-specific PH for biomolecules),
Adams et al. 2017 (persistence images). The novelty TOPOLOGIX claims is NOT
this vanilla pipeline — it is the interface-restricted bipartite construct in
`complex`/`metric`; this module is the shared, citable PH backend that both the
ligand-only (track B) and interface (track C) features run through.

Design choices for reproducibility (honesty rail: stable features):
- Fixed birth/persistence ranges and pixel size, so the vector dimension and
  scale are independent of the training set (no fit-on-train leakage).
- H0 (connected components) -> histogram of finite death scales. H0 births are
  all 0 under VR, so a persistence *image* would be degenerate; the death-scale
  histogram captures the merge structure instead.
- H1/H2 (loops/voids) -> persistence images (persim), plus per-dimension summary
  stats (count, max/sum/mean persistence) that are robust when a diagram is sparse.
"""
from __future__ import annotations

import numpy as np

# Fixed filtration window (Angstrom). Covers the bonded/contact scale where
# molecular loops and voids form; features beyond this are dominated by global
# size, which descriptors already capture.
_BIRTH_RANGE = (0.0, 8.0)
_PERS_RANGE = (0.0, 8.0)
_PIXEL = 1.0          # -> 8x8 = 64 px per imaged dimension
_H0_BINS = 16         # death-scale histogram resolution for H0
_H0_MAX = 8.0


def vr_persistence(points: np.ndarray, maxdim: int = 1) -> dict[int, np.ndarray]:
    """Vietoris-Rips persistence of a point cloud.

    points : (n_atoms, 3) coordinates.
    Returns {dim: (n_features, 2) birth/death array} for dim in 0..maxdim.
    Infinite deaths are dropped (the single essential H0 class, and any unclosed
    higher class within the threshold).
    """
    from ripser import ripser
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}
    dgms = ripser(pts, maxdim=maxdim)["dgms"]
    out = {}
    for d in range(maxdim + 1):
        dg = dgms[d] if d < len(dgms) else np.empty((0, 2))
        dg = np.asarray(dg, dtype=float)
        if dg.size:
            dg = dg[np.isfinite(dg).all(axis=1)]
        out[d] = dg.reshape(-1, 2)
    return out


def _persistence_image(dgm: np.ndarray):
    """Flatten one birth/death diagram to a fixed-length persistence image vector."""
    from persim import PersistenceImager
    imgr = PersistenceImager(birth_range=_BIRTH_RANGE, pers_range=_PERS_RANGE,
                             pixel_size=_PIXEL)
    if dgm.shape[0] == 0:
        # empty diagram -> zero image of the canonical shape
        nb = int(round((_BIRTH_RANGE[1] - _BIRTH_RANGE[0]) / _PIXEL))
        npx = int(round((_PERS_RANGE[1] - _PERS_RANGE[0]) / _PIXEL))
        return np.zeros(nb * npx, dtype=float)
    img = imgr.transform(dgm, skew=True)
    return np.asarray(img, dtype=float).ravel()


def _summary_stats(dgm: np.ndarray) -> np.ndarray:
    """Robust per-diagram summary: [count, max_pers, sum_pers, mean_pers]."""
    if dgm.shape[0] == 0:
        return np.zeros(4, dtype=float)
    pers = dgm[:, 1] - dgm[:, 0]
    return np.array([len(pers), pers.max(), pers.sum(), pers.mean()], dtype=float)


def _h0_histogram(dgm: np.ndarray) -> np.ndarray:
    """Histogram of H0 finite death scales (component-merge structure)."""
    if dgm.shape[0] == 0:
        return np.zeros(_H0_BINS, dtype=float)
    deaths = np.clip(dgm[:, 1], 0.0, _H0_MAX)
    hist, _ = np.histogram(deaths, bins=_H0_BINS, range=(0.0, _H0_MAX))
    return hist.astype(float)


def vectorize(diagrams: dict[int, np.ndarray], maxdim: int = 1) -> np.ndarray:
    """Diagrams -> single fixed-length feature vector.

    Layout: H0 death histogram + summary(H0) ; then for each d>=1:
    persistence image(d) + summary(d). Length is constant for a given maxdim.
    """
    parts = [_h0_histogram(diagrams.get(0, np.empty((0, 2)))),
             _summary_stats(diagrams.get(0, np.empty((0, 2))))]
    for d in range(1, maxdim + 1):
        dg = diagrams.get(d, np.empty((0, 2)))
        parts.append(_persistence_image(dg))
        parts.append(_summary_stats(dg))
    return np.concatenate(parts)


def ph_vector(points: np.ndarray, maxdim: int = 1) -> np.ndarray:
    """Convenience: point cloud -> PH feature vector."""
    return vectorize(vr_persistence(points, maxdim=maxdim), maxdim=maxdim)
