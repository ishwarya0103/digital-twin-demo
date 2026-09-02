"""Writes rows to the audit log (models.py). Every Phase 1-5 pipeline function calls
`log_audit_event()` once at the end of its own run -- a single added call, not a duplication of
that function's logic -- so every pipeline invocation across the whole system is captured in one
place for post hoc review, per the architecture doc's Section 7 "Audit & traceability" concern.
"""

from sqlalchemy.orm import Session

from src.db.session import SessionLocal, engine
from src.governance.models import AuditLogEntry


def log_audit_event(
    pipeline_stage: str,
    action: str,
    patient_id: str | None = None,
    source_file: str | None = None,
    pipeline_version: str | None = None,
    db: Session | None = None,
) -> AuditLogEntry:
    """Writes one audit log row and returns it.

    If `db` isn't given, opens and commits its own short-lived session against the project's
    real database (`src.db.session.SessionLocal`) -- so every pipeline function can call this
    without every caller up the chain needing to thread a db session through just for logging.
    Pass `db` explicitly (tests; a caller that already has an open session, e.g.
    `assemble_and_store_twin`) to write into that session/transaction instead.

    Also lazily ensures its own table exists on that fallback path -- callers using the real
    database (rather than a test fixture that already runs `Base.metadata.create_all()` for
    every registered table) shouldn't have to separately provision `audit_log` before their
    first pipeline run.
    """
    owns_session = db is None
    if owns_session:
        AuditLogEntry.__table__.create(bind=engine, checkfirst=True)
    session = db or SessionLocal()
    try:
        entry = AuditLogEntry(
            patient_id=patient_id,
            pipeline_stage=pipeline_stage,
            action=action,
            source_file=source_file,
            pipeline_version=pipeline_version,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    finally:
        if owns_session:
            session.close()
