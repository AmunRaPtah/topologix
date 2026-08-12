"""Provenance guard tests.

These exist because of a real near-miss: installing PyTDC to fetch a split silently
downgraded rdkit, changing `Descriptors._descList` from 217 entries to 210 and so changing
the descriptor baseline mid-experiment. Nothing in the pre-registration, the gate, or the
analysis would have caught it. The first test below reconstructs exactly that situation.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import provenance  # noqa: E402


def _write(tmp_path, name, **overrides):
    rec = {
        "python": "3.12.3",
        "platform": "test",
        "versions": {"rdkit": "2026.03.5", "numpy": "2.5.1"},
        "block_dims": {"A": 217, "V": 224},
        "source": "herg_karim.tab",
        "source_sha256": "b47cbff0" * 8,
        "rdkit_descriptor_count": 217,
    }
    rec.update(overrides)
    features = tmp_path / name
    features.write_bytes(b"")
    with open(provenance.sidecar_path(str(features)), "w") as fh:
        json.dump(rec, fh)
    return str(features)


def test_matching_environments_pass(tmp_path):
    a = _write(tmp_path, "a.pkl")
    b = _write(tmp_path, "b.pkl")
    provenance.assert_compatible(a, b)


def test_descriptor_count_change_is_caught(tmp_path):
    """The actual failure: rdkit downgrade moved the descriptor block from 217 to 210."""
    a = _write(tmp_path, "a.pkl")
    b = _write(tmp_path, "b.pkl",
               versions={"rdkit": "2023.09.6", "numpy": "2.5.1"},
               block_dims={"A": 210, "V": 224},
               rdkit_descriptor_count=210)
    with pytest.raises(provenance.ProvenanceMismatch) as exc:
        provenance.assert_compatible(a, b)
    assert "block_dims" in str(exc.value)
    assert "versions.rdkit" in str(exc.value)


def test_different_source_data_is_caught(tmp_path):
    a = _write(tmp_path, "a.pkl")
    b = _write(tmp_path, "b.pkl", source_sha256="deadbeef" * 8)
    with pytest.raises(provenance.ProvenanceMismatch):
        provenance.assert_compatible(a, b)


def test_missing_sidecar_is_a_failure(tmp_path):
    """An unprovenanced matrix cannot be shown comparable, so it is not assumed to be."""
    a = _write(tmp_path, "a.pkl")
    orphan = tmp_path / "orphan.pkl"
    orphan.write_bytes(b"")
    with pytest.raises(provenance.ProvenanceMismatch, match="no provenance sidecar"):
        provenance.assert_compatible(a, str(orphan))


def test_non_blocking_difference_passes(tmp_path):
    """A numpy patch bump is recorded but does not block; only BLOCKING fields do."""
    a = _write(tmp_path, "a.pkl")
    b = _write(tmp_path, "b.pkl", versions={"rdkit": "2026.03.5", "numpy": "2.5.2"})
    provenance.assert_compatible(a, b)


def test_capture_records_live_descriptor_count():
    rec = provenance.capture()
    assert rec["versions"]["rdkit"] is not None
    assert isinstance(rec["rdkit_descriptor_count"], int)
