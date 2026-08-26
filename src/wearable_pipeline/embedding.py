"""Stage 5 (embedding generation): produces a patient's wearable-derived physiological state
profile -- the longitudinal ouch-meter trajectory (each window's activation score, in order)
plus a fixed-length embedding summarizing it, mean-pooled from the LSTM's per-window hidden
states (the same representation the activation score was read off of, so the embedding and
the score are grounded in the same evidence)."""

import numpy as np

from src.wearable_pipeline.model import HIDDEN_SIZE
from src.wearable_pipeline.models import WearableProfile, WearableWindow

EMBEDDING_DIM = HIDDEN_SIZE


def build_wearable_profile(
    windows: list[WearableWindow], patient_id: str, pipeline_version: str
) -> WearableProfile:
    ordered = sorted(windows, key=lambda w: w.start_time)

    hidden_states = [w.hidden_state for w in ordered if w.hidden_state is not None]
    if hidden_states:
        embedding = np.mean(hidden_states, axis=0).astype(np.float32)
    else:
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    return WearableProfile(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        windows=ordered,
        embedding=embedding,
    )
