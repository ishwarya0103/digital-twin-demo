"""Turns a patient's wearable-derived physiological state profile (Stage 5's `WearableProfile`)
into a labeled, plain-language summary -- the actual stress/pain-activation score and a simple
interpretation -- for anything that needs clinical language rather than the opaque 16-dim
hidden-state embedding. Reuses each window's own `activation_score` (Stage 4); does not
recompute or re-score anything.
"""

import numpy as np

from src.wearable_pipeline.models import WearableProfile

# Fixed thresholds over the [0, 1] activation score the LSTM's sigmoid output is already
# bounded to -- not a validated clinical severity scale. The "ouch meter" is trained against a
# heuristic autonomic-activation proxy, not real patient-reported pain/symptom labels (see
# PROGRESS.md's Known Issues); treat this interpretation label as illustrative, same caveat.
_LOW_THRESHOLD = 0.33
_HIGH_THRESHOLD = 0.66


def _interpret(mean_score: float) -> str:
    if mean_score >= _HIGH_THRESHOLD:
        return "high autonomic/stress activation"
    if mean_score >= _LOW_THRESHOLD:
        return "moderate autonomic/stress activation"
    return "low autonomic/stress activation"


def summarize_wearable_profile(profile: WearableProfile) -> dict:
    """{"mean_activation_score", "max_activation_score", "num_windows", "interpretation"}."""
    scores = [w.activation_score for w in profile.windows if w.activation_score is not None]

    if not scores:
        return {
            "mean_activation_score": None,
            "max_activation_score": None,
            "num_windows": 0,
            "interpretation": "no scored wearable windows available",
        }

    mean_score = float(np.mean(scores))
    return {
        "mean_activation_score": mean_score,
        "max_activation_score": float(np.max(scores)),
        "num_windows": len(scores),
        "interpretation": _interpret(mean_score),
    }
