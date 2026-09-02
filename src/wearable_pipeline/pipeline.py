from pathlib import Path

from sqlalchemy.orm import Session

from src.governance.audit import log_audit_event
from src.wearable_pipeline.embedding import build_wearable_profile
from src.wearable_pipeline.feature_extraction import extract_window_features
from src.wearable_pipeline.model import score_window, train_model
from src.wearable_pipeline.models import WearableProfile, WearableWindow
from src.wearable_pipeline.preprocessing import preprocess_session
from src.wearable_pipeline.segmentation import build_event_windows
from src.wearable_pipeline.signal_loader import discover_sessions, discover_subjects, load_session

PIPELINE_VERSION = "wearable-v0.1.0"


def build_windows_for_subject(
    raw_dir, subject: str, pipeline_version: str = PIPELINE_VERSION
) -> list[WearableWindow]:
    subject_dir = Path(raw_dir) / subject
    windows: list[WearableWindow] = []

    for session_name in discover_sessions(subject_dir):
        raw_session = load_session(subject_dir, session_name, subject, pipeline_version)
        session = preprocess_session(raw_session)

        for start, end, window_type in build_event_windows(session):
            features = extract_window_features(session, start, end)
            windows.append(
                WearableWindow(
                    patient_id=subject,
                    pipeline_version=pipeline_version,
                    session=session_name,
                    window_type=window_type,
                    start_time=start,
                    end_time=end,
                    feature_sequence=features,
                )
            )

    return windows


def run_wearable_pipeline(
    raw_dir: str = "data/raw/wearable",
    pipeline_version: str = PIPELINE_VERSION,
    epochs: int = 150,
    seed: int = 0,
    db: Session | None = None,
) -> list[WearableProfile]:
    subjects = discover_subjects(raw_dir)

    all_windows: list[WearableWindow] = []
    for subject in subjects:
        all_windows.extend(build_windows_for_subject(raw_dir, subject, pipeline_version))

    # One small model shared across the whole cohort (per the architecture doc: "the same
    # time-series model, trained against patient-reported labels"), not one per patient.
    model, mean, std = train_model(all_windows, epochs=epochs, seed=seed)
    for window in all_windows:
        score, hidden = score_window(model, mean, std, window.feature_sequence)
        window.activation_score = score
        window.hidden_state = hidden

    profiles = [
        build_wearable_profile(
            [w for w in all_windows if w.patient_id == subject], subject, pipeline_version
        )
        for subject in subjects
    ]

    for subject in subjects:
        log_audit_event(
            pipeline_stage="wearable_pipeline",
            action="run_wearable_pipeline",
            patient_id=subject,
            source_file=str(Path(raw_dir) / subject),
            pipeline_version=pipeline_version,
            db=db,
        )

    return profiles
