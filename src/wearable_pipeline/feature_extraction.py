"""Stage 3 (feature extraction): HRV and other autonomic-proxy features via NeuroKit2, one
feature vector per sub-interval of an event window -- the sequence the LSTM (stage 4) consumes.

HRV comes from peak-detecting the raw BVP/PPG signal directly (`nk.ppg_process` +
`nk.hrv_time`) rather than from Empatica's onboard IBI.csv extraction: IBI.csv turns out to be
extremely sparse for this dataset (median gaps of ~2s between detected beats, with gaps of
minutes), too gap-ridden for reliable interval statistics, whereas NeuroKit2's own PPG peak
detector runs on the continuous raw waveform and finds beats it can actually use.
"""

import warnings

import neurokit2 as nk
import numpy as np

from src.wearable_pipeline.models import SessionSignals

STEP_WIDTH_SECONDS = 60.0

FEATURE_NAMES = (
    "hrv_mean_nn", "hrv_sdnn", "hrv_rmssd",
    "eda_tonic_mean", "eda_phasic_std", "eda_scr_count",
    "temp_mean", "temp_std",
    "hr_mean", "hr_std",
)
NUM_FEATURES = len(FEATURE_NAMES)

_MIN_SECONDS_FOR_HRV = 5.0
_MIN_SECONDS_FOR_EDA = 5.0


def _slice_channel(channel: np.ndarray, rate: float, channel_start: float, start: float, end: float) -> np.ndarray:
    start_idx = max(0, int(round((start - channel_start) * rate)))
    end_idx = min(len(channel), int(round((end - channel_start) * rate)))
    if end_idx <= start_idx:
        return np.array([])
    return channel[start_idx:end_idx]


def _hrv_features(bvp_slice: np.ndarray, sample_rate: float) -> tuple[float, float, float]:
    if len(bvp_slice) < sample_rate * _MIN_SECONDS_FOR_HRV:
        return 0.0, 0.0, 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, info = nk.ppg_process(bvp_slice, sampling_rate=int(sample_rate))
            if len(info.get("PPG_Peaks", [])) < 3:
                return 0.0, 0.0, 0.0
            hrv = nk.hrv_time(info, sampling_rate=int(sample_rate))
        row = hrv.iloc[0]
        values = (row.get("HRV_MeanNN", 0.0), row.get("HRV_SDNN", 0.0), row.get("HRV_RMSSD", 0.0))
        return tuple(0.0 if v is None or np.isnan(v) else float(v) for v in values)
    except Exception:
        return 0.0, 0.0, 0.0


def _eda_features(eda_slice: np.ndarray, sample_rate: float) -> tuple[float, float, float]:
    if len(eda_slice) < sample_rate * _MIN_SECONDS_FOR_EDA:
        return 0.0, 0.0, 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals, info = nk.eda_process(eda_slice, sampling_rate=int(sample_rate))
        tonic_mean = float(np.nan_to_num(signals["EDA_Tonic"].mean()))
        phasic_std = float(np.nan_to_num(signals["EDA_Phasic"].std()))
        scr_count = float(len(info.get("SCR_Peaks", [])))
        return tonic_mean, phasic_std, scr_count
    except Exception:
        return 0.0, 0.0, 0.0


def _simple_stats(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def extract_window_features(
    session: SessionSignals, start: float, end: float, step_width: float = STEP_WIDTH_SECONDS
) -> np.ndarray:
    """Returns (num_steps, NUM_FEATURES); at least one step even for windows shorter than
    step_width."""
    step_starts = np.arange(start, end, step_width)
    if len(step_starts) == 0:
        step_starts = np.array([start])

    rows = []
    for step_start in step_starts:
        step_end = min(step_start + step_width, end)

        bvp_slice = _slice_channel(session.bvp, session.bvp_rate, session.bvp_start, step_start, step_end)
        eda_slice = _slice_channel(session.eda, session.eda_rate, session.eda_start, step_start, step_end)
        temp_slice = _slice_channel(session.temp, session.temp_rate, session.temp_start, step_start, step_end)
        hr_slice = _slice_channel(session.hr, session.hr_rate, session.hr_start, step_start, step_end)

        hrv_mean_nn, hrv_sdnn, hrv_rmssd = _hrv_features(bvp_slice, session.bvp_rate)
        eda_tonic_mean, eda_phasic_std, eda_scr_count = _eda_features(eda_slice, session.eda_rate)
        temp_mean, temp_std = _simple_stats(temp_slice)
        hr_mean, hr_std = _simple_stats(hr_slice)

        rows.append([
            hrv_mean_nn, hrv_sdnn, hrv_rmssd,
            eda_tonic_mean, eda_phasic_std, eda_scr_count,
            temp_mean, temp_std,
            hr_mean, hr_std,
        ])

    return np.array(rows, dtype=np.float32)
