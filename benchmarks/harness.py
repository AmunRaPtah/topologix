"""Benchmark harness: the validation gate.

Every derivation issue is validated here. The gate that turns the
mathematical core into defensible IP is:

    IPC-derived features must beat a molecular-descriptor baseline
    (RDKit descriptors + XGBoost) on a held-out hERG split.

If they do not, the formulation under test is wrong and the
dependency chain halts at the validation issue rather than shipping
a platform that only looks finished.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GateResult:
    method_auc: float
    baseline_auc: float

    @property
    def passes(self) -> bool:
        return self.method_auc > self.baseline_auc


def run_gate(method_scores, baseline_scores, y_true) -> GateResult:
    """Compare IPC method vs descriptor baseline by ROC AUC.

    To be implemented by the validation issue. Raises until then so a
    skipped or stubbed gate fails loudly rather than passing silently.
    """
    raise NotImplementedError("Implement in validation issue PRI-7 gate.")
