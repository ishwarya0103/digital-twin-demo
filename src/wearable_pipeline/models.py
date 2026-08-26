from dataclasses import dataclass, field

import numpy as np


@dataclass
class SessionSignals:
    """Raw (preprocessed) per-channel signals for one exam session, still on their own
    per-channel sampling rates and physical units."""

    patient_id: str
    pipeline_version: str
    session: str  # "Final" | "Midterm 1" | "Midterm 2"
    hr: np.ndarray
    hr_rate: float
    hr_start: float  # each channel gets its own start time -- Empatica's HR channel starts
    eda: np.ndarray  # ~10s after the others (its onboard averaging needs a warm-up buffer)
    eda_rate: float
    eda_start: float
    bvp: np.ndarray
    bvp_rate: float
    bvp_start: float
    temp: np.ndarray
    temp_rate: float
    temp_start: float
    tags: list[float]  # unix timestamps of any labeled events; empty if none recorded


@dataclass
class WearableWindow:
    """One event window (onset/escalation/resolution) with its feature sequence and,
    once scored, its pain-activation output -- the unit the "ouch meter" trajectory is
    built from."""

    patient_id: str
    pipeline_version: str
    session: str
    window_type: str  # "tagged" | "onset" | "resolution"
    start_time: float
    end_time: float
    feature_sequence: np.ndarray  # (num_steps, num_features)
    activation_score: float | None = None
    hidden_state: np.ndarray | None = None  # LSTM final hidden state, for the embedding stage


@dataclass
class WearableProfile:
    """Stage 5 output: a patient's wearable-derived physiological state profile --
    the longitudinal ouch-meter trajectory plus a fixed-length embedding summarizing it."""

    patient_id: str
    pipeline_version: str
    windows: list[WearableWindow]
    embedding: np.ndarray
