"""FastAPI backend (architecture doc: read access to the digital twin store and the fusion
layer's generated hypotheses). Retrospective, advisory-only -- these endpoints only ever read
already-computed, already-stored data; nothing here runs a pipeline, calls the Claude API, or
performs any reasoning/prediction on request.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.schemas import DomainEmbedding, DomainResponse, FullTwinResponse, HypothesesResponse, PatientsResponse
from src.db.base import Base
from src.db.session import engine, get_db
from src.digital_twin.retrieval import DOMAINS, get_twin, get_twin_domain, list_patient_ids
from src.fusion_layer.retrieval import get_hypotheses_for_patient


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Every model in the project shares this one declarative Base (src/db/base.py), so this
    # creates every table -- digital_twins, hypotheses, audit_log -- that doesn't exist yet,
    # the same way each phase's own test fixtures and verification scripts already do.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Digital Twin Demo API", lifespan=lifespan)


@app.get("/health")
def health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.get("/patients", response_model=PatientsResponse)
def list_patients(db: Session = Depends(get_db)):
    """Every patient_id with at least one stored digital twin -- what the demo UI's patient
    picker (app/main.py) populates itself from."""
    return PatientsResponse(patient_ids=list_patient_ids(db))


@app.get("/patient/{patient_id}/full-twin", response_model=FullTwinResponse)
def get_full_twin(patient_id: str, db: Session = Depends(get_db)):
    twin = get_twin(db, patient_id)
    if twin is None:
        raise HTTPException(status_code=404, detail=f"No digital twin found for patient_id={patient_id!r}")

    return FullTwinResponse(
        patient_id=twin.patient_id,
        version=twin.version,
        created_at=twin.created_at,
        emr=DomainEmbedding(
            pipeline_version=twin.emr_pipeline_version, embedding=twin.emr_embedding, summary=twin.emr_summary or {}
        ),
        genomic=DomainEmbedding(
            pipeline_version=twin.genomic_pipeline_version,
            embedding=twin.genomic_embedding,
            summary=twin.genomic_summary or {},
        ),
        wearable=DomainEmbedding(
            pipeline_version=twin.wearable_pipeline_version,
            embedding=twin.wearable_embedding,
            summary=twin.wearable_summary or {},
        ),
    )


def _get_domain_or_404(patient_id: str, domain: str, db: Session) -> DomainResponse:
    result = get_twin_domain(db, patient_id, domain)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No digital twin found for patient_id={patient_id!r}")
    return DomainResponse(**result)


@app.get("/patient/{patient_id}/emr", response_model=DomainResponse)
def get_emr_domain(patient_id: str, db: Session = Depends(get_db)):
    return _get_domain_or_404(patient_id, "emr", db)


@app.get("/patient/{patient_id}/genomic", response_model=DomainResponse)
def get_genomic_domain(patient_id: str, db: Session = Depends(get_db)):
    return _get_domain_or_404(patient_id, "genomic", db)


@app.get("/patient/{patient_id}/wearable", response_model=DomainResponse)
def get_wearable_domain(patient_id: str, db: Session = Depends(get_db)):
    return _get_domain_or_404(patient_id, "wearable", db)


assert set(DOMAINS) == {"emr", "genomic", "wearable"}, "a domain route above is missing/stale"


@app.get("/patient/{patient_id}/hypotheses", response_model=HypothesesResponse)
def get_hypotheses(patient_id: str, db: Session = Depends(get_db)):
    """The fusion layer's stored hypotheses that cite this patient, each with its
    source_embedding_ids (the traceability references back to the exact embeddings that
    produced it) included. An empty list is a valid 200, not a 404 -- a patient having no
    hypotheses yet isn't an error."""
    hypotheses = get_hypotheses_for_patient(db, patient_id)
    return HypothesesResponse(patient_id=patient_id, hypotheses=hypotheses)
