"""Element-specific bipartite interface persistent homology.

Geometry is carried over unchanged from /root/topologix/benchmark/src/esph_interface.py:
same 8 A interface cut, same 16 A filtration cap, same fixed-range persistence imager with
its range fixed BEFORE fitting (no train-set leakage into the vectorizer), same six
element-pair channels. Only the input path differs: ligand poses arrive from SDF here
rather than from HETATM records in a PDB.

Keeping the geometry identical is deliberate. If this passes where the Platinum version
failed, the difference must be attributable to the endpoint, which is what invariant I-3
predicts, and not to a re-tuned featurizer.
"""

import numpy as np
from persim import PersistenceImager
from ripser import ripser

CUT = 8.0        # interface = protein heavy atoms within CUT of any ligand heavy atom
BIG = 1e6        # within-affiliation distance: bipartite, only cross links exist
THRESH = 16.0    # ripser filtration cap on the opposition-distance complex

_BIRTH_RANGE = (0.0, 16.0)
_PERS_RANGE = (0.0, 16.0)
_PIXEL = 4.0
_H0_BINS = 8
_H0_MAX = 16.0
_IMGR = PersistenceImager(birth_range=_BIRTH_RANGE, pers_range=_PERS_RANGE, pixel_size=_PIXEL)
_IMG_LEN = (int(round((_BIRTH_RANGE[1] - _BIRTH_RANGE[0]) / _PIXEL))
            * int(round((_PERS_RANGE[1] - _PERS_RANGE[0]) / _PIXEL)))

CHANNELS = [("ALL", "ALL"), ("ALL", "C"), ("ALL", "N"), ("ALL", "O"),
            (("N", "O"), "ALL"), (("N", "O"), ("N", "O"))]

_CHAN_LEN = _H0_BINS + 4 + _IMG_LEN + 4
VECTOR_LEN = _CHAN_LEN * len(CHANNELS)


def _summary(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(4)
    p = dgm[:, 1] - dgm[:, 0]
    return np.array([len(p), p.max(), p.sum(), p.mean()])


def _h0hist(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(_H0_BINS)
    h, _ = np.histogram(np.clip(dgm[:, 1], 0, _H0_MAX), bins=_H0_BINS, range=(0, _H0_MAX))
    return h.astype(float)


def _img(dgm):
    if dgm.shape[0] == 0:
        return np.zeros(_IMG_LEN)
    return np.asarray(_IMGR.transform(dgm, skew=True), float).ravel()


def vectorize_dm(M):
    if M is None or M.shape[0] < 2:
        return np.zeros(_CHAN_LEN)
    dgms = ripser(M, distance_matrix=True, maxdim=1, thresh=THRESH)["dgms"]
    d0 = dgms[0]
    d0 = d0[np.isfinite(d0).all(1)] if d0.size else d0.reshape(-1, 2)
    d1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    d1 = d1[np.isfinite(d1).all(1)] if d1.size else d1.reshape(-1, 2)
    return np.concatenate([_h0hist(d0), _summary(d0), _img(d1), _summary(d1)])


def opp_dm(lig, inter):
    nl = len(lig)
    n = nl + len(inter)
    M = np.full((n, n), BIG)
    cross = np.sqrt(((lig[:, None, :] - inter[None, :, :]) ** 2).sum(-1))
    M[:nl, nl:] = cross
    M[nl:, :nl] = cross.T
    np.fill_diagonal(M, 0.0)
    return M


def _sel(coords, elems, which):
    if which == "ALL":
        return coords
    keep = np.isin(elems, list(which) if isinstance(which, tuple) else [which])
    return coords[keep]


def esph_interface_vector(lig, lig_e, inter, inter_e):
    parts = []
    for le, pe in CHANNELS:
        L = _sel(lig, lig_e, le)
        P = _sel(inter, inter_e, pe)
        if len(L) < 2 or len(P) < 2:
            parts.append(np.zeros(_CHAN_LEN))
            continue
        parts.append(vectorize_dm(opp_dm(L, P)))
    return np.concatenate(parts)


def parse_protein(path):
    """Heavy-atom coordinates and element symbols from ATOM records."""
    coords, elems = [], []
    with open(path) as fh:
        for line in fh:
            if line[:6].strip() != "ATOM":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            e = line[76:78].strip()
            if not (e and e.isalpha()):
                e = (line[12:16].strip() or "C")[0]
            e = e.capitalize()
            if e == "H":
                continue
            coords.append((x, y, z))
            elems.append(e)
    return np.array(coords, float), np.array(elems)


def interface_of(lig, prot, prot_e):
    """Protein heavy atoms within CUT of any ligand heavy atom."""
    if len(lig) == 0 or len(prot) == 0:
        return None
    d = np.sqrt(((prot[:, None, :] - lig[None, :, :]) ** 2).sum(-1))
    m = d.min(1) <= CUT
    if m.sum() < 4:
        return None
    return prot[m], prot_e[m]
