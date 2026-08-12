"""Environment provenance for cached feature matrices.

Exists because of a specific near-miss, recorded in benchmarks/README.md: installing PyTDC
to fetch a split silently downgraded rdkit, which changed `Descriptors._descList` from 217
entries to 210 and so changed the descriptor baseline mid-experiment. Nothing in the
pre-registration, the gate, or the analysis would have caught it. It was noticed only
because a dimension printed differently between two runs.

The fix is not a note in a README. A feature matrix carries a sidecar recording the library
versions, the block dimensions, and a hash of the source data that produced it, and any code
comparing two matrices calls `assert_compatible` and fails loudly on disagreement.

    from provenance import capture, write_sidecar, assert_compatible

    write_sidecar("results/features.pkl", blocks={"A": 217, ...}, source=SRC)
    ...
    assert_compatible("results/features.pkl", "results/features_run1.pkl")
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

# Libraries whose version can silently change a feature matrix. rdkit is first because it
# is the one that actually did.
TRACKED = ["rdkit", "numpy", "scipy", "sklearn", "xgboost", "ripser", "persim", "gudhi"]

# Fields that must match for two feature matrices to be comparable. Version differences in
# anything outside this set are recorded but do not block.
BLOCKING = ["block_dims", "source_sha256", "versions.rdkit"]


def _version(mod_name: str) -> str | None:
    try:
        mod = __import__(mod_name)
    except Exception:
        return None
    for attr in ("__version__", "version"):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
    return "present, version unknown"


def _sha256(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture(block_dims: dict | None = None, source: str | None = None) -> dict:
    """Snapshot everything that could change a feature matrix without anyone noticing."""
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {name: _version(name) for name in TRACKED},
        "block_dims": dict(block_dims or {}),
        "source": source,
        "source_sha256": _sha256(source) if source else None,
    }
    # The descriptor list length is the specific quantity that moved, so record it directly
    # rather than inferring it from the rdkit version.
    try:
        from rdkit.Chem import Descriptors
        rec["rdkit_descriptor_count"] = len(Descriptors._descList)
    except Exception:
        rec["rdkit_descriptor_count"] = None
    return rec


def sidecar_path(features_path: str) -> str:
    return os.path.splitext(features_path)[0] + ".provenance.json"


def write_sidecar(features_path: str, block_dims: dict | None = None,
                  source: str | None = None) -> str:
    rec = capture(block_dims, source)
    path = sidecar_path(features_path)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    return path


def read_sidecar(features_path: str) -> dict | None:
    path = sidecar_path(features_path)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _get(rec: dict, dotted: str):
    cur = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


class ProvenanceMismatch(RuntimeError):
    pass


def assert_compatible(*features_paths: str, blocking=BLOCKING) -> dict:
    """Raise unless every named feature matrix was produced by a comparable environment.

    A missing sidecar is itself a failure: an unprovenanced matrix cannot be shown to be
    comparable, and 'probably fine' is what this module exists to stop.
    """
    recs = {}
    for p in features_paths:
        rec = read_sidecar(p)
        if rec is None:
            raise ProvenanceMismatch(
                f"{p}: no provenance sidecar. Regenerate it with write_sidecar(), or "
                "establish by hand that the environment matches before comparing."
            )
        recs[p] = rec

    ref_path, ref = next(iter(recs.items()))
    problems = []
    for p, rec in list(recs.items())[1:]:
        for field in blocking:
            a, b = _get(ref, field), _get(rec, field)
            if a != b:
                problems.append(f"  {field}\n    {ref_path}: {a}\n    {p}: {b}")
    if problems:
        raise ProvenanceMismatch(
            "feature matrices are not comparable:\n" + "\n".join(problems)
            + "\n\nThis is the check that would have caught the rdkit 217 -> 210 descriptor "
              "change. Do not compare these results; regenerate one under the other's "
              "environment, or state the difference explicitly in the write-up."
        )
    return ref


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2, sort_keys=True))
