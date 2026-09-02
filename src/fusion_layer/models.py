"""Layer 3 -- Generative Semantic Fusion Layer (architecture doc Section 5). Storage for the
structured hypotheses this layer generates. Per the doc's "Traceability layer" component
("Links every generated hypothesis back to the specific embedding evidence that produced it"),
every row's `source_embedding_ids` names the exact chromadb entries (see vector_store.py's
`embedding_id()`) -- and therefore the exact DigitalTwin rows and domains -- that produced it.
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String

from src.db.base import Base


class Hypothesis(Base):
    """One row per generated hypothesis. Never updated in place -- each call to
    generate_and_store_hypothesis() (reasoning.py) inserts a new row."""

    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    model = Column(String, nullable=False)  # Claude model id that generated this hypothesis

    # The strict, predefined schema fields (see reasoning.py's HYPOTHESIS_JSON_SCHEMA).
    subgroup_trait = Column(String, nullable=False)
    supporting_evidence = Column(JSON, nullable=False)  # list[str]
    confidence = Column(Float, nullable=False)
    # embedding_id()-formatted strings ("{patient_id}:v{version}:{domain}") -- the traceability
    # link back to the exact source embeddings (DigitalTwin rows + domains) behind this
    # hypothesis. Always non-empty: a hypothesis with no source embedding is not traceable.
    source_embedding_ids = Column(JSON, nullable=False)
