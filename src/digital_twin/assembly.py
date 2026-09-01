"""Assembles the three Layer 1 domain embeddings for a patient into a new, versioned
DigitalTwin row (architecture doc Section 4). Pure assembly and storage only -- no
reasoning, prediction, or transformation of the embeddings themselves; converting a numpy
vector to a JSON-storable list is serialization, not a data transformation.
"""

import numpy as np
from sqlalchemy.orm import Session

from src.digital_twin.models import DigitalTwin


def _as_list(embedding: np.ndarray) -> list[float]:
    return np.asarray(embedding, dtype=np.float64).tolist()


def _next_version(db: Session, patient_id: str) -> int:
    latest = (
        db.query(DigitalTwin)
        .filter(DigitalTwin.patient_id == patient_id)
        .order_by(DigitalTwin.version.desc())
        .first()
    )
    return 1 if latest is None else latest.version + 1


def assemble_and_store_twin(
    db: Session,
    patient_id: str,
    emr_embedding: np.ndarray,
    emr_pipeline_version: str,
    genomic_embedding: np.ndarray,
    genomic_pipeline_version: str,
    wearable_embedding: np.ndarray,
    wearable_pipeline_version: str,
) -> DigitalTwin:
    """Writes a new versioned digital twin row for `patient_id`. Never overwrites or deletes
    a previous version for that patient -- `version` is always one more than that patient's
    current highest version (1 for a first twin)."""
    twin = DigitalTwin(
        patient_id=patient_id,
        version=_next_version(db, patient_id),
        emr_pipeline_version=emr_pipeline_version,
        emr_embedding=_as_list(emr_embedding),
        genomic_pipeline_version=genomic_pipeline_version,
        genomic_embedding=_as_list(genomic_embedding),
        wearable_pipeline_version=wearable_pipeline_version,
        wearable_embedding=_as_list(wearable_embedding),
    )
    db.add(twin)
    db.commit()
    db.refresh(twin)
    return twin
