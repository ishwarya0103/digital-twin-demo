"""Stage 5 (Embedding generation): encodes a patient's chronological timeline into a
fixed-length clinical state vector, Deep-Patient style -- mean-pooling per-event
Bio_ClinicalBERT embeddings over the whole timeline. Note-derived events reuse the
contextual embedding computed during extraction (stage 3); structured events get one
computed here from a templated "{event_type}: {text}" string, since they never passed
through free text.
"""

import numpy as np

from src.emr_pipeline.models import ClinicalEvent
from src.emr_pipeline.nlp_extraction import embed_text

EMBEDDING_DIM = 768


def generate_clinical_state_vector(
    timeline: list[ClinicalEvent], patient_id: str, pipeline_version: str
) -> np.ndarray:
    if not timeline:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    vectors = []
    for event in timeline:
        assert event.patient_id == patient_id
        assert event.pipeline_version == pipeline_version
        vector = event.context_embedding
        if vector is None:
            vector = embed_text(f"{event.event_type}: {event.text}")
        vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32).mean(axis=0)
