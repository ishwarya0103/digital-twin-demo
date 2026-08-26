"""Reads one subject/session's raw Empatica E4 export (Wearable Exam Stress Dataset format).

Expected input layout: data/raw/wearable/<subject>/<session>/{HR,EDA,BVP,TEMP,IBI,ACC,
tags}.csv, exactly as distributed -- single-column channels (HR, EDA, BVP, TEMP) have the
session start unix timestamp on line 1 and the sample rate in Hz on line 2, followed by one
reading per line; tags.csv holds zero or more unix timestamps of physical button-press
events and is often empty. IBI.csv (Empatica's onboard beat detector) turns out to be too
sparse to compute reliable HRV from directly (see feature_extraction.py) so it isn't loaded
here; BVP is used instead as the basis for peak detection.
"""

from pathlib import Path

import numpy as np

from src.wearable_pipeline.models import SessionSignals

_CHANNEL_FILES = ("HR.csv", "EDA.csv", "BVP.csv", "TEMP.csv")
SESSION_NAMES = ("Final", "Midterm 1", "Midterm 2")


def _load_channel(path: Path) -> tuple[float, float, np.ndarray]:
    with open(path) as f:
        start_time = float(f.readline())
        sample_rate = float(f.readline())
    values = np.loadtxt(path, delimiter=",", skiprows=2)
    return start_time, sample_rate, values


def _load_tags(path: Path) -> list[float]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    values = np.loadtxt(path, delimiter=",", ndmin=1)
    return sorted(float(v) for v in values)


def discover_subjects(raw_dir) -> list[str]:
    raw_dir = Path(raw_dir)
    return sorted(p.name for p in raw_dir.iterdir() if p.is_dir())


def discover_sessions(subject_dir) -> list[str]:
    subject_dir = Path(subject_dir)
    return [s for s in SESSION_NAMES if (subject_dir / s).is_dir()]


def load_session(
    subject_dir, session: str, patient_id: str, pipeline_version: str
) -> SessionSignals:
    session_dir = Path(subject_dir) / session
    for filename in _CHANNEL_FILES:
        if not (session_dir / filename).exists():
            raise ValueError(f"{session_dir} is missing {filename}")

    hr_start, hr_rate, hr = _load_channel(session_dir / "HR.csv")
    eda_start, eda_rate, eda = _load_channel(session_dir / "EDA.csv")
    bvp_start, bvp_rate, bvp = _load_channel(session_dir / "BVP.csv")
    temp_start, temp_rate, temp = _load_channel(session_dir / "TEMP.csv")
    tags = _load_tags(session_dir / "tags.csv")

    return SessionSignals(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        session=session,
        hr=hr,
        hr_rate=hr_rate,
        hr_start=hr_start,
        eda=eda,
        eda_rate=eda_rate,
        eda_start=eda_start,
        bvp=bvp,
        bvp_rate=bvp_rate,
        bvp_start=bvp_start,
        temp=temp,
        temp_rate=temp_rate,
        temp_start=temp_start,
        tags=tags,
    )


def load_all_sessions(subject_dir, patient_id: str, pipeline_version: str) -> list[SessionSignals]:
    return [
        load_session(subject_dir, session, patient_id, pipeline_version)
        for session in discover_sessions(subject_dir)
    ]
