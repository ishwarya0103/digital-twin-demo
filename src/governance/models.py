"""Cross-cutting governance layer (architecture doc Section 7): "Audit & traceability |
Every transformation and generative output is logged for post hoc review | Audit logging /
lineage tracking". `AuditLogEntry` is that log: one row per pipeline run, across every phase.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from src.db.base import Base


class AuditLogEntry(Base):
    """One row per pipeline invocation. `patient_id` is nullable because a handful of actions
    are inherently cohort-level rather than about one patient (see audit.py)."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    patient_id = Column(String, nullable=True, index=True)
    pipeline_stage = Column(String, nullable=False)
    action = Column(String, nullable=False)
    source_file = Column(String, nullable=True)
    pipeline_version = Column(String, nullable=True)
