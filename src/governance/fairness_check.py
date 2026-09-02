"""Governance check stub (architecture doc Section 7, "Bias monitoring | Model performance and
subgroup stability evaluated across age, sex, ancestry, severity strata | Fairlearn, AI Fairness
360"). Demonstrates the *mechanism* -- a metric computed per subgroup via fairlearn's
MetricFrame, plus the spread across subgroups -- not a validated bias finding.

At this project's 5-patient scale, any subgroup split is definitionally too small for a
fairness statistic to mean anything: one patient's value alone can swing an entire subgroup's
mean. `n_per_group` and `small_sample_warning` below make that visible rather than hiding it.
Real usage means wiring in real outcome labels/predictions and real demographic strata (age,
sex, ancestry, severity, per the doc) once cohort size is large enough for the numbers to be
statistically meaningful -- this stub exists to show where and how that would plug in, per the
doc's framing of bias monitoring as a cross-cutting governance concern, not an optional add-on.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from fairlearn.metrics import MetricFrame

# A conventional rule-of-thumb floor for a subgroup mean to start being statistically stable.
# This demo's 5-patient cohort never comes close -- every real check here will warn.
MIN_MEANINGFUL_GROUP_SIZE = 30


@dataclass
class SubgroupFairnessResult:
    metric_by_group: dict[str, float]
    n_per_group: dict[str, int]
    max_difference: float
    max_ratio: float
    small_sample_warning: bool


def _mean_metric(y_true, y_pred) -> float:
    """No ground-truth outcome labels exist in this demo (see module docstring) -- `y_true` is
    accepted only because fairlearn's MetricFrame requires one, and is otherwise ignored."""
    return float(np.mean(y_pred))


def check_subgroup_fairness(
    y_pred: list[float], sensitive_features: list, y_true: list[float] | None = None
) -> SubgroupFairnessResult:
    """Computes the mean of `y_pred` per subgroup of `sensitive_features` (fairlearn's
    MetricFrame), plus the max difference/ratio across subgroups -- fairlearn's standard
    subgroup-disparity summary. Pass a real `y_true` and a real scoring metric once real
    outcome labels exist; until then this reports on `y_pred`'s own distribution across
    groups, enough to demonstrate the mechanism without pretending to score anything against
    ground truth that doesn't exist here."""
    if len(y_pred) != len(sensitive_features):
        raise ValueError("y_pred and sensitive_features must be the same length")

    y_true = y_true if y_true is not None else y_pred

    frame = MetricFrame(metrics=_mean_metric, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
    n_per_group = Counter(sensitive_features)

    return SubgroupFairnessResult(
        metric_by_group={str(k): float(v) for k, v in frame.by_group.items()},
        n_per_group={str(k): v for k, v in n_per_group.items()},
        max_difference=float(frame.difference()),
        max_ratio=float(frame.ratio()),
        small_sample_warning=any(n < MIN_MEANINGFUL_GROUP_SIZE for n in n_per_group.values()),
    )
