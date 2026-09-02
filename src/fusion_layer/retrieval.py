"""Read access to stored hypotheses (models.py) -- selects and filters already-stored rows,
no reasoning happens here.
"""

from sqlalchemy.orm import Session

from src.fusion_layer.models import Hypothesis


def get_hypotheses_for_patient(db: Session, patient_id: str) -> list[Hypothesis]:
    """Every hypothesis that cites at least one of `patient_id`'s embeddings in its
    `source_embedding_ids`. There's no `patient_id` column on `Hypothesis` (one hypothesis can
    span several patients' embeddings -- see models.py), so this matches by parsing each
    stored ID's leading `"{patient_id}:..."` component (the same convention
    `src.fusion_layer.formatting.embedding_id()` produces and `reasoning.store_hypothesis()`
    parses for audit logging) rather than a database-level JSON query, keeping this portable
    across SQLite and Postgres without dialect-specific JSON operators."""
    matches = []
    for hypothesis in db.query(Hypothesis).order_by(Hypothesis.created_at.desc()).all():
        source_patient_ids = {eid.split(":")[0] for eid in hypothesis.source_embedding_ids}
        if patient_id in source_patient_ids:
            matches.append(hypothesis)
    return matches
