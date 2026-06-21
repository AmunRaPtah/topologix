"""Constraint set for the PH backend + ligand-PH featurizer (track B).

These are the invariants the math must satisfy regardless of the gate outcome:
fixed-length output, rotation/translation invariance, and graceful handling of
degenerate / invalid inputs.
"""
import numpy as np
import pytest

from topologix.homology import ph_vector, vr_persistence
from topologix.features import conformer_cloud, ligand_ph_features

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


def test_vector_fixed_length_regardless_of_atom_count():
    a = ph_vector(conformer_cloud(ASPIRIN), maxdim=1)
    b = ph_vector(conformer_cloud(CAFFEINE), maxdim=1)
    assert a.shape == b.shape
    assert a.shape == (88,)  # 16 H0 hist + 4 H0 stats + 64 H1 image + 4 H1 stats


def test_rotation_translation_invariance():
    cloud = conformer_cloud(ASPIRIN)
    v1 = ph_vector(cloud, maxdim=1)
    rng = np.random.RandomState(0)
    Q, _ = np.linalg.qr(rng.randn(3, 3))      # random rotation/reflection
    v2 = ph_vector(cloud @ Q + np.array([10.0, -5.0, 3.0]), maxdim=1)
    # VR PH depends only on pairwise distances -> exact invariance
    assert np.allclose(v1, v2, atol=1e-9)


def test_perturbation_stability():
    """Small coordinate noise -> small feature change (persistence-image stability)."""
    cloud = conformer_cloud(ASPIRIN)
    v1 = ph_vector(cloud, maxdim=1)
    rng = np.random.RandomState(1)
    v2 = ph_vector(cloud + rng.normal(scale=0.05, size=cloud.shape), maxdim=1)
    assert np.linalg.norm(v1 - v2) < np.linalg.norm(v1)  # bounded, not blown up


def test_degenerate_clouds():
    assert vr_persistence(np.zeros((1, 3)))[0].shape == (0, 2)   # single point
    assert vr_persistence(np.zeros((0, 3)))[1].shape == (0, 2)   # empty


def test_featurizer_mask_drops_invalid_and_aligns():
    X, mask = ligand_ph_features([ASPIRIN, "not_a_smiles", CAFFEINE, "C"], maxdim=1)
    assert mask.tolist() == [True, False, True, False]  # bad parse + 1-heavy-atom dropped
    assert X.shape == (2, 88)
    assert np.isfinite(X).all()
