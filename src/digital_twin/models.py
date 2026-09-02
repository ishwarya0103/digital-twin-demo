"""Layer 2 -- Digital Twin Abstraction Layer (architecture doc Section 4).

The digital twin is the concatenation of the three independent Layer 1 embeddings for a
single patient -- Clinical State Vector (EMR), Genomic Pathway Profile, and Wearable-Derived
Physiological State Profile -- with no reasoning, prediction, or transformation applied. Per
the doc's Layer 2 design-choice table:
  - Update frequency: batch/retrospective, not real-time streaming
  - Traceability: every twin component remains traceable to its source domain and pipeline
    version
  - Scope boundary: no closed-loop control; advisory/analytic outputs only
  - Storage format: structured embedding record per patient (versioned), not raw data
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Integer, String, UniqueConstraint

from src.db.base import Base


class DigitalTwin(Base):
    """One row per patient per version. Never updated or deleted in place -- a new patient
    state is always a new row with an incremented `version`, so every prior twin snapshot
    for a patient remains queryable exactly as it was assembled."""

    __tablename__ = "digital_twins"
    __table_args__ = (UniqueConstraint("patient_id", "version", name="uq_digital_twin_patient_version"),)

    id = Column(Integer, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # Each domain's embedding is stored alongside the pipeline_version that produced it, per
    # the doc's traceability requirement -- the column names make the source domain explicit,
    # a plain "embedding" column would lose that labeling.
    emr_pipeline_version = Column(String, nullable=False)
    emr_embedding = Column(JSON, nullable=False)  # list[float] -- Stage 5 clinical state vector
    # Labeled clinical summary alongside the opaque embedding (src/emr_pipeline/summary.py's
    # summarize_clinical_events() output: {"diagnoses", "medications", "symptoms"}, each a
    # list[str]) -- already-extracted, already-structured Stage 3/4 output, not raw note text.
    # Nullable so a twin assembled before this column existed still loads; every twin assembled
    # since always has one (assembly.py requires it).
    emr_summary = Column(JSON, nullable=True)

    genomic_pipeline_version = Column(String, nullable=False)
    genomic_embedding = Column(JSON, nullable=False)  # list[float] -- genomic pathway profile
    # src/genomics_pipeline/summary.py's summarize_pathway_profile() output:
    # {"pathway_scores": {name: score}, "ancestry_pcs": list[float]}.
    genomic_summary = Column(JSON, nullable=True)

    wearable_pipeline_version = Column(String, nullable=False)
    wearable_embedding = Column(JSON, nullable=False)  # list[float] -- wearable physiological profile
    # src/wearable_pipeline/summary.py's summarize_wearable_profile() output:
    # {"mean_activation_score", "max_activation_score", "num_windows", "interpretation"}.
    wearable_summary = Column(JSON, nullable=True)
