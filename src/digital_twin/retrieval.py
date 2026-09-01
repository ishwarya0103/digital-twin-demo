"""Read access to stored digital twins (architecture doc Section 4) -- the full,
concatenated per-patient state, or just one domain's slice of it. No reasoning or
transformation happens here either: this only selects and reshapes already-stored rows.
"""

from sqlalchemy.orm import Session

from src.digital_twin.models import DigitalTwin

DOMAINS = ("emr", "genomic", "wearable")


def get_twin(db: Session, patient_id: str, version: int | None = None) -> DigitalTwin | None:
    """The full twin record for a patient: the latest version by default, or a specific
    `version` if given. None if no twin has been assembled for this patient (and version,
    if given)."""
    query = db.query(DigitalTwin).filter(DigitalTwin.patient_id == patient_id)
    if version is not None:
        return query.filter(DigitalTwin.version == version).first()
    return query.order_by(DigitalTwin.version.desc()).first()


def get_twin_domain(db: Session, patient_id: str, domain: str, version: int | None = None) -> dict | None:
    """Just one domain's slice of a patient's twin -- its embedding and the pipeline_version
    that produced it, nothing from the other two domains. `domain` is one of DOMAINS
    ("emr", "genomic", "wearable"). None if no matching twin exists."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}, expected one of {DOMAINS}")

    twin = get_twin(db, patient_id, version)
    if twin is None:
        return None

    return {
        "patient_id": twin.patient_id,
        "version": twin.version,
        "domain": domain,
        "pipeline_version": getattr(twin, f"{domain}_pipeline_version"),
        "embedding": getattr(twin, f"{domain}_embedding"),
    }
