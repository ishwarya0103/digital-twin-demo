"""End-to-end test for src/digital_twin/orchestrate.py: runs the real EMR, wearable, and
genomics pipelines (no mocking, same convention as tests/test_{emr,wearable,genomics}_pipeline.py)
for one patient and confirms a correctly assembled, three-domain twin comes out the other end.

The three source IDs below are one arbitrary, explicit correspondence across the three
datasets' otherwise-unrelated ID spaces (see orchestrate.py's module docstring) -- not a claim
that this Synthea patient, this wearable subject, and this VCF sample are "the same person".
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.digital_twin.orchestrate import orchestrate_twin_for_patient
from src.digital_twin.retrieval import get_twin
from src.emr_pipeline import PIPELINE_VERSION as EMR_PIPELINE_VERSION
from src.emr_pipeline.embedding import EMBEDDING_DIM as EMR_EMBEDDING_DIM
from src.genomics_pipeline import PIPELINE_VERSION as GENOMICS_PIPELINE_VERSION
from src.wearable_pipeline import PIPELINE_VERSION as WEARABLE_PIPELINE_VERSION
from src.wearable_pipeline.embedding import EMBEDDING_DIM as WEARABLE_EMBEDDING_DIM

# Del587_Abernathy524_d98c0bff-46fc-7b08-a6a6-2fca67f0ab0b.json's Patient.id
EMR_PATIENT_ID = "d98c0bff-46fc-7b08-a6a6-2fca67f0ab0b"
WEARABLE_SUBJECT_ID = "S1"
GENOMIC_SAMPLE_ID = "HG00096"


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_orchestrate_twin_for_patient_end_to_end(db):
    twin = orchestrate_twin_for_patient(
        db,
        patient_id="patient-1",
        emr_patient_id=EMR_PATIENT_ID,
        wearable_subject_id=WEARABLE_SUBJECT_ID,
        genomic_sample_id=GENOMIC_SAMPLE_ID,
    )

    assert twin.patient_id == "patient-1"
    assert twin.version == 1

    assert twin.emr_pipeline_version == EMR_PIPELINE_VERSION
    assert len(twin.emr_embedding) == EMR_EMBEDDING_DIM
    assert any(v != 0.0 for v in twin.emr_embedding)  # this patient actually has clinical events

    assert twin.wearable_pipeline_version == WEARABLE_PIPELINE_VERSION
    assert len(twin.wearable_embedding) == WEARABLE_EMBEDDING_DIM

    assert twin.genomic_pipeline_version == GENOMICS_PIPELINE_VERSION
    assert len(twin.genomic_embedding) > 0

    # Round-trip through storage/retrieval, not just the in-memory object returned above.
    fetched = get_twin(db, "patient-1")
    assert fetched.id == twin.id
    assert fetched.emr_embedding == twin.emr_embedding
    assert fetched.wearable_embedding == twin.wearable_embedding
    assert fetched.genomic_embedding == twin.genomic_embedding
