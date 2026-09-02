import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from src.db.base import Base
from src.db.session import get_db
from src.digital_twin.assembly import assemble_and_store_twin
from src.fusion_layer.formatting import embedding_id
from src.fusion_layer.reasoning import store_hypothesis

EMR_EMBEDDING = [1.0, 2.0, 3.0]
GENOMIC_EMBEDDING = [4.0, 5.0]
WEARABLE_EMBEDDING = [6.0, 7.0, 8.0]

EMR_SUMMARY = {"diagnoses": ["Fibromyalgia"], "medications": ["Metformin"], "symptoms": ["back pain"]}
GENOMIC_SUMMARY = {"pathway_scores": {"Drug metabolism": 1.5}, "ancestry_pcs": [0.1, -0.2]}
WEARABLE_SUMMARY = {
    "mean_activation_score": 0.42,
    "max_activation_score": 0.6,
    "num_windows": 3,
    "interpretation": "moderate autonomic/stress activation",
}


@pytest.fixture
def db_session():
    """A fresh in-memory SQLite database per test -- isolated from the project's real
    data/processed/twin.db, which is gitignored and empty on a fresh checkout anyway.

    check_same_thread=False + StaticPool: FastAPI's TestClient dispatches sync route handlers
    on a worker thread, not the test's own thread, and a bare in-memory SQLite connection is
    both thread-bound and, per connection, its own separate empty database -- StaticPool keeps
    the whole engine on a single shared connection so the seeded data in this fixture's thread
    is still there when a request handler reads it from another.
    """
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def seeded_twin(db_session):
    """One digital twin with three domains carrying distinct, individually-recognizable
    embeddings -- so a test can positively confirm the *right* domain's data comes back, not
    just that *some* data comes back."""
    return assemble_and_store_twin(
        db_session,
        patient_id="patient-test",
        emr_embedding=EMR_EMBEDDING,
        emr_pipeline_version="emr-v0.1.0",
        emr_summary=EMR_SUMMARY,
        genomic_embedding=GENOMIC_EMBEDDING,
        genomic_pipeline_version="genomics-v0.1.0",
        genomic_summary=GENOMIC_SUMMARY,
        wearable_embedding=WEARABLE_EMBEDDING,
        wearable_pipeline_version="wearable-v0.1.0",
        wearable_summary=WEARABLE_SUMMARY,
    )


@pytest.fixture
def seeded_hypothesis(db_session, seeded_twin):
    hypothesis = {
        "subgroup_trait": "test subgroup trait",
        "supporting_evidence": ["observation one", "observation two"],
        "confidence": 0.42,
        "source_embedding_ids": [
            embedding_id("patient-test", 1, "emr"),
            embedding_id("patient-test", 1, "genomic"),
        ],
    }
    return store_hypothesis(db_session, hypothesis, model="claude-sonnet-5")


# ---------------------------------------------------------------------------
# GET /health, /patients
# ---------------------------------------------------------------------------


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_patients_returns_200_with_seeded_patient(client, seeded_twin):
    response = client.get("/patients")
    assert response.status_code == 200
    assert response.json() == {"patient_ids": ["patient-test"]}


# ---------------------------------------------------------------------------
# GET /patient/{id}/full-twin
# ---------------------------------------------------------------------------


def test_full_twin_returns_200_with_correctly_shaped_data(client, seeded_twin):
    response = client.get("/patient/patient-test/full-twin")
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"patient_id", "version", "created_at", "emr", "genomic", "wearable"}
    assert data["patient_id"] == "patient-test"
    assert data["version"] == 1

    for domain_key, embedding, pipeline_version, summary in (
        ("emr", EMR_EMBEDDING, "emr-v0.1.0", EMR_SUMMARY),
        ("genomic", GENOMIC_EMBEDDING, "genomics-v0.1.0", GENOMIC_SUMMARY),
        ("wearable", WEARABLE_EMBEDDING, "wearable-v0.1.0", WEARABLE_SUMMARY),
    ):
        assert set(data[domain_key].keys()) == {"pipeline_version", "embedding", "summary"}
        assert data[domain_key]["embedding"] == embedding
        assert data[domain_key]["pipeline_version"] == pipeline_version
        assert data[domain_key]["summary"] == summary


def test_full_twin_returns_404_for_unknown_patient(client):
    response = client.get("/patient/no-such-patient/full-twin")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /patient/{id}/{emr,genomic,wearable}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain,expected_embedding,expected_pipeline_version,expected_summary",
    [
        ("emr", EMR_EMBEDDING, "emr-v0.1.0", EMR_SUMMARY),
        ("genomic", GENOMIC_EMBEDDING, "genomics-v0.1.0", GENOMIC_SUMMARY),
        ("wearable", WEARABLE_EMBEDDING, "wearable-v0.1.0", WEARABLE_SUMMARY),
    ],
)
def test_domain_endpoint_returns_200_with_correctly_shaped_data(
    client, seeded_twin, domain, expected_embedding, expected_pipeline_version, expected_summary
):
    response = client.get(f"/patient/patient-test/{domain}")
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"patient_id", "version", "domain", "pipeline_version", "embedding", "summary"}
    assert data["patient_id"] == "patient-test"
    assert data["version"] == 1
    assert data["domain"] == domain
    assert data["pipeline_version"] == expected_pipeline_version
    assert data["embedding"] == expected_embedding
    assert data["summary"] == expected_summary


@pytest.mark.parametrize(
    "domain,other_embeddings,other_pipeline_versions,other_summaries",
    [
        (
            "emr",
            [GENOMIC_EMBEDDING, WEARABLE_EMBEDDING],
            ["genomics-v0.1.0", "wearable-v0.1.0"],
            [GENOMIC_SUMMARY, WEARABLE_SUMMARY],
        ),
        (
            "genomic",
            [EMR_EMBEDDING, WEARABLE_EMBEDDING],
            ["emr-v0.1.0", "wearable-v0.1.0"],
            [EMR_SUMMARY, WEARABLE_SUMMARY],
        ),
        (
            "wearable",
            [EMR_EMBEDDING, GENOMIC_EMBEDDING],
            ["emr-v0.1.0", "genomics-v0.1.0"],
            [EMR_SUMMARY, GENOMIC_SUMMARY],
        ),
    ],
)
def test_domain_endpoint_never_includes_other_domains(
    client, seeded_twin, domain, other_embeddings, other_pipeline_versions, other_summaries
):
    response = client.get(f"/patient/patient-test/{domain}")
    data = response.json()

    # No key named after another domain (e.g. the /emr response has no "genomic"/"wearable" key).
    for other_domain in {"emr", "genomic", "wearable"} - {domain}:
        assert other_domain not in data

    # The single embedding/pipeline_version/summary field never carries another domain's values.
    for other_embedding in other_embeddings:
        assert data["embedding"] != other_embedding
    for other_pipeline_version in other_pipeline_versions:
        assert data["pipeline_version"] != other_pipeline_version
    for other_summary in other_summaries:
        assert data["summary"] != other_summary


def test_domain_endpoint_returns_404_for_unknown_patient(client):
    response = client.get("/patient/no-such-patient/genomic")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /patient/{id}/hypotheses
# ---------------------------------------------------------------------------


def test_hypotheses_endpoint_returns_200_with_source_references(client, seeded_hypothesis):
    response = client.get("/patient/patient-test/hypotheses")
    assert response.status_code == 200

    data = response.json()
    assert data["patient_id"] == "patient-test"
    assert len(data["hypotheses"]) == 1

    hypothesis = data["hypotheses"][0]
    assert set(hypothesis.keys()) == {
        "id",
        "created_at",
        "model",
        "subgroup_trait",
        "supporting_evidence",
        "confidence",
        "source_embedding_ids",
    }
    assert hypothesis["subgroup_trait"] == "test subgroup trait"
    assert hypothesis["source_embedding_ids"] == [
        embedding_id("patient-test", 1, "emr"),
        embedding_id("patient-test", 1, "genomic"),
    ]


def test_hypotheses_endpoint_returns_200_with_empty_list_for_patient_with_none(client, seeded_twin):
    """A patient with no hypotheses yet is a valid empty result, not a 404."""
    response = client.get("/patient/patient-test/hypotheses")
    assert response.status_code == 200
    assert response.json() == {"patient_id": "patient-test", "hypotheses": []}
