"""Benchmark harness: the validation gate.

Every derivation issue is validated here. The gate that turns the mathematical core into
defensible IP (per docs/BUILD_PLAN.md, locked decision — topology as an orthogonal booster):

    Topological features must add SIGNIFICANT signal on top of the strong descriptor baseline
    (RDKit descriptors + XGBoost) on a held-out hERG split — judged on AUROC and, especially,
    MCC (where descriptors plateau on imbalanced data). "Significant" = a paired-bootstrap 95%
    CI on the metric delta that excludes zero. A raw point gain that is within noise does NOT
    pass.

If the booster does not clear the gate, the formulation under test is wrong and the dependency
chain halts at validation rather than shipping a platform that only looks finished. A failed
gate is a logged result that redirects the build, not a setback to paper over.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GateResult:
    """Outcome of comparing a candidate representation against the baseline."""
    metric: str
    delta: float
    ci_lo: float
    ci_hi: float
    baseline_value: float
    method_value: float
    detail: dict = field(default_factory=dict)

    @property
    def passes(self) -> bool:
        # significant improvement = the whole 95% CI of (method - baseline) is above 0
        return self.ci_lo > 0.0

    def __str__(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (f"[{verdict}] {self.metric}: baseline={self.baseline_value:.4f} "
                f"method={self.method_value:.4f}  delta={self.delta:+.4f} "
                f"(95% CI [{self.ci_lo:+.4f}, {self.ci_hi:+.4f}], "
                f"p(method>baseline)={self.detail.get('p_b_gt_a', float('nan')):.2f})")


def run_gate(method_scores, baseline_scores, y_true, metric: str = "auroc",
             n_boot: int = 2000, seed: int = 0) -> GateResult:
    """Compare a candidate's test scores vs the baseline's on the SAME held-out rows.

    method_scores / baseline_scores : predicted positive-class probabilities, aligned to y_true.
    Significance via paired bootstrap of the metric delta (metrics.bootstrap_delta).
    """
    from topologix.metrics import bootstrap_delta, classification_metrics
    bv = classification_metrics(y_true, baseline_scores)[metric]
    mv = classification_metrics(y_true, method_scores)[metric]
    bs = bootstrap_delta(y_true, baseline_scores, method_scores, metric=metric,
                         n_boot=n_boot, seed=seed)
    return GateResult(metric=metric, delta=bs["delta"], ci_lo=bs["ci_lo"], ci_hi=bs["ci_hi"],
                      baseline_value=bv, method_value=mv, detail=bs)
