import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.digital_twin.assembly import assemble_and_store_twin
from src.digital_twin.retrieval import get_twin, get_twin_domain


@pytest.fixture
def db():
    """A fresh in-memory SQLite database per test -- isolated from the project's real
    data/processed/twin.db."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


PATIENT_ID = "patient-001"

EMR_SUMMARY_V1 = {"diagnoses": ["Fibromyalgia"], "medications": ["Metformin"], "symptoms": ["back pain"]}
GENOMIC_SUMMARY_V1 = {"pathway_scores": {"Drug metabolism": 1.5}, "ancestry_pcs": [0.1, -0.2]}
WEARABLE_SUMMARY_V1 = {
    "mean_activation_score": 0.42,
    "max_activation_score": 0.6,
    "num_windows": 3,
    "interpretation": "moderate autonomic/stress activation",
}


def _twin_v1(db):
    return assemble_and_store_twin(
        db,
        patient_id=PATIENT_ID,
        emr_embedding=np.array([1.0, 2.0, 3.0]),
        emr_pipeline_version="emr-v0.1.0",
        emr_summary=EMR_SUMMARY_V1,
        genomic_embedding=np.array([4.0, 5.0]),
        genomic_pipeline_version="genomics-v0.1.0",
        genomic_summary=GENOMIC_SUMMARY_V1,
        wearable_embedding=np.array([6.0, 7.0, 8.0, 9.0]),
        wearable_pipeline_version="wearable-v0.1.0",
        wearable_summary=WEARABLE_SUMMARY_V1,
    )


def test_stored_twin_contains_all_three_embeddings_labeled_by_source(db):
    twin = _twin_v1(db)

    assert twin.emr_embedding == [1.0, 2.0, 3.0]
    assert twin.emr_pipeline_version == "emr-v0.1.0"

    assert twin.genomic_embedding == [4.0, 5.0]
    assert twin.genomic_pipeline_version == "genomics-v0.1.0"

    assert twin.wearable_embedding == [6.0, 7.0, 8.0, 9.0]
    assert twin.wearable_pipeline_version == "wearable-v0.1.0"

    # Round-trip through storage and back via get_twin, not just the in-memory object.
    fetched = get_twin(db, PATIENT_ID)
    assert fetched.emr_embedding == [1.0, 2.0, 3.0]
    assert fetched.genomic_embedding == [4.0, 5.0]
    assert fetched.wearable_embedding == [6.0, 7.0, 8.0, 9.0]


def test_stored_twin_contains_labeled_clinical_summary_per_domain(db):
    twin = _twin_v1(db)

    assert twin.emr_summary == EMR_SUMMARY_V1
    assert twin.genomic_summary == GENOMIC_SUMMARY_V1
    assert twin.wearable_summary == WEARABLE_SUMMARY_V1

    # Round-trips through storage, not just the in-memory object.
    fetched = get_twin(db, PATIENT_ID)
    assert fetched.emr_summary == EMR_SUMMARY_V1
    assert fetched.genomic_summary == GENOMIC_SUMMARY_V1
    assert fetched.wearable_summary == WEARABLE_SUMMARY_V1


def test_second_version_does_not_delete_or_alter_first(db):
    v1 = _twin_v1(db)
    assert v1.version == 1

    v2 = assemble_and_store_twin(
        db,
        patient_id=PATIENT_ID,
        emr_embedding=np.array([10.0, 20.0, 30.0]),
        emr_pipeline_version="emr-v0.2.0",
        emr_summary={"diagnoses": ["Essential hypertension"], "medications": [], "symptoms": []},
        genomic_embedding=np.array([40.0, 50.0]),
        genomic_pipeline_version="genomics-v0.1.0",
        genomic_summary=GENOMIC_SUMMARY_V1,
        wearable_embedding=np.array([60.0, 70.0, 80.0, 90.0]),
        wearable_pipeline_version="wearable-v0.1.0",
        wearable_summary=WEARABLE_SUMMARY_V1,
    )
    assert v2.version == 2

    # The first version is still present and untouched.
    fetched_v1 = get_twin(db, PATIENT_ID, version=1)
    assert fetched_v1 is not None
    assert fetched_v1.emr_embedding == [1.0, 2.0, 3.0]
    assert fetched_v1.emr_pipeline_version == "emr-v0.1.0"
    assert fetched_v1.emr_summary == EMR_SUMMARY_V1

    fetched_v2 = get_twin(db, PATIENT_ID, version=2)
    assert fetched_v2.emr_embedding == [10.0, 20.0, 30.0]
    assert fetched_v2.emr_pipeline_version == "emr-v0.2.0"
    assert fetched_v2.emr_summary == {"diagnoses": ["Essential hypertension"], "medications": [], "symptoms": []}

    # Both rows genuinely exist side by side.
    assert db.query(type(v1)).filter_by(patient_id=PATIENT_ID).count() == 2

    # Default (no version given) retrieval returns the latest.
    assert get_twin(db, PATIENT_ID).version == 2


def test_domain_filtered_retrieval_returns_only_requested_domain(db):
    _twin_v1(db)

    genomic_only = get_twin_domain(db, PATIENT_ID, "genomic")
    assert genomic_only["domain"] == "genomic"
    assert genomic_only["embedding"] == [4.0, 5.0]
    assert genomic_only["pipeline_version"] == "genomics-v0.1.0"
    assert genomic_only["summary"] == GENOMIC_SUMMARY_V1
    assert "emr_embedding" not in genomic_only
    assert "wearable_embedding" not in genomic_only
    assert set(genomic_only.keys()) == {"patient_id", "version", "domain", "pipeline_version", "embedding", "summary"}

    emr_only = get_twin_domain(db, PATIENT_ID, "emr")
    assert emr_only["embedding"] == [1.0, 2.0, 3.0]
    assert emr_only["pipeline_version"] == "emr-v0.1.0"
    assert emr_only["summary"] == EMR_SUMMARY_V1

    wearable_only = get_twin_domain(db, PATIENT_ID, "wearable")
    assert wearable_only["embedding"] == [6.0, 7.0, 8.0, 9.0]
    assert wearable_only["pipeline_version"] == "wearable-v0.1.0"
    assert wearable_only["summary"] == WEARABLE_SUMMARY_V1


def test_domain_filtered_retrieval_rejects_unknown_domain(db):
    _twin_v1(db)
    with pytest.raises(ValueError):
        get_twin_domain(db, PATIENT_ID, "not_a_real_domain")


def test_get_twin_returns_none_for_unknown_patient(db):
    assert get_twin(db, "no-such-patient") is None
    assert get_twin_domain(db, "no-such-patient", "emr") is None
