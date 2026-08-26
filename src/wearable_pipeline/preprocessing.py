"""Stage 1 (signal preprocessing): noise filtering and normalization, NumPy/SciPy only.

Filtering happens on the raw, physically-scaled signal (a Butterworth filter per channel,
cutoffs chosen for what each sensor actually measures) so downstream NeuroKit2 processing --
which expects real physiological units to compute PPG peak amplitudes, EDA tonic/phasic
levels, etc. -- still sees a physically sensible signal. Normalization is applied later, to
the per-window *feature* vectors the sequence model consumes, rather than to the raw signal
itself: z-scoring raw EDA microsiemens or PPG amplitude before NeuroKit2 sees them would
distort the physical scale its algorithms rely on.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from src.wearable_pipeline.models import SessionSignals

# (kind, cutoff(s) in Hz) per channel. BVP gets a bandpass matching the typical pulse band;
# EDA/TEMP/HR are slow-changing signals, so a low lowpass cutoff removes sensor noise without
# smearing the physiological trend.
FILTER_SPECS = {
    "bvp": ("bandpass", (0.5, 8.0)),
    "eda": ("lowpass", 1.0),
    "temp": ("lowpass", 0.1),
    "hr": ("lowpass", 0.3),
}


def filter_signal(signal: np.ndarray, sample_rate: float, channel: str, order: int = 4) -> np.ndarray:
    if len(signal) < order * 3 + 1:
        return signal  # too short to filter safely; leave as-is

    kind, cutoff = FILTER_SPECS[channel]
    nyquist = sample_rate / 2
    if kind == "bandpass":
        low, high = cutoff
        wn = (low / nyquist, min(high / nyquist, 0.99))
        b, a = butter(order, wn, btype="band")
    else:
        wn = min(cutoff / nyquist, 0.99)
        b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, signal)


def preprocess_session(session: SessionSignals) -> SessionSignals:
    return SessionSignals(
        patient_id=session.patient_id,
        pipeline_version=session.pipeline_version,
        session=session.session,
        hr=filter_signal(session.hr, session.hr_rate, "hr"),
        hr_rate=session.hr_rate,
        hr_start=session.hr_start,
        eda=filter_signal(session.eda, session.eda_rate, "eda"),
        eda_rate=session.eda_rate,
        eda_start=session.eda_start,
        bvp=filter_signal(session.bvp, session.bvp_rate, "bvp"),
        bvp_rate=session.bvp_rate,
        bvp_start=session.bvp_start,
        temp=filter_signal(session.temp, session.temp_rate, "temp"),
        temp_rate=session.temp_rate,
        temp_start=session.temp_start,
        tags=session.tags,
    )


def fit_normalizer(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean/std for z-scoring. Fit once on the whole cohort's feature rows so the
    same scale applies consistently at both training and inference time."""
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_normalizer(feature_matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (feature_matrix - mean) / std
