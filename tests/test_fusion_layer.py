from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import jsonschema
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.digital_twin.assembly import assemble_and_store_twin
from src.fusion_layer.clustering import cluster_twins
from src.fusion_layer.formatting import embedding_id, format_cluster_summary, format_patient_summary
from src.fusion_layer.models import Hypothesis
from src.fusion_layer.reasoning import (
    HYPOTHESIS_JSON_SCHEMA,
    _call_claude,
    generate_and_store_hypothesis,
    generate_hypothesis,
    store_hypothesis,
)
from src.fusion_layer.vector_store import get_chroma_client, query_similar, upsert_twin_embeddings


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _make_twin(db, patient_id: str, emr_offset: float, genomic_offset: float, wearable_offset: float):
    """A digital twin with embeddings offset from zero by a fixed amount -- lets clustering
    tests build two clearly-separable groups without needing real pipeline output."""
    return assemble_and_store_twin(
        db,
        patient_id=patient_id,
        emr_embedding=[emr_offset] * 8,
        emr_pipeline_version="emr-v0.1.0",
        genomic_embedding=[genomic_offset] * 5,
        genomic_pipeline_version="genomics-v0.1.0",
        wearable_embedding=[wearable_offset] * 4,
        wearable_pipeline_version="wearable-v0.1.0",
    )


VALID_HYPOTHESIS = {
    "subgroup_trait": "Elevated inflammatory pathway signal with suppressed HRV",
    "supporting_evidence": [
        "Genomic domain mean per-dimension value is markedly higher than the cohort baseline",
        "Wearable domain L2 norm is elevated relative to the other cluster",
    ],
    "confidence": 0.62,
    "source_embedding_ids": [],  # filled in per-test with real candidate IDs
}


def _mock_client_returning(hypothesis: dict) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    tool_use_block = SimpleNamespace(type="tool_use", input=hypothesis, id="toolu_fake", name="record_hypothesis")
    client.messages.create.return_value = SimpleNamespace(content=[tool_use_block])
    return client


# ---------------------------------------------------------------------------
# formatting.py
# ---------------------------------------------------------------------------


def test_format_patient_summary_labels_every_domain_no_raw_values(db):
    twin = _make_twin(db, "patient-1", 1.0, 2.0, 3.0)
    text = format_patient_summary(twin)

    for domain in ("emr", "genomic", "wearable"):
        assert domain in text
        assert embedding_id("patient-1", 1, domain) in text
    # Structured summary statistics only -- never the embedding's raw value list.
    assert str(twin.emr_embedding) not in text
    assert str(twin.genomic_embedding) not in text
    assert str(twin.wearable_embedding) not in text


def test_format_cluster_summary_includes_every_member(db):
    twin_a = _make_twin(db, "patient-1", 1.0, 1.0, 1.0)
    twin_b = _make_twin(db, "patient-2", 1.1, 1.1, 1.1)
    text = format_cluster_summary([twin_a, twin_b], cluster_id=0)

    assert "patient-1" in text
    assert "patient-2" in text
    assert "Cluster 0" in text


# ---------------------------------------------------------------------------
# clustering.py
# ---------------------------------------------------------------------------


def test_cluster_twins_groups_similar_patients_together(db):
    low_a = _make_twin(db, "low-1", 0.0, 0.0, 0.0)
    low_b = _make_twin(db, "low-2", 0.1, -0.1, 0.05)
    high_a = _make_twin(db, "high-1", 50.0, 50.0, 50.0)
    high_b = _make_twin(db, "high-2", 49.9, 50.1, 49.8)

    clusters = cluster_twins([low_a, low_b, high_a, high_b], n_clusters=2)

    assert len(clusters) == 2
    member_ids_per_cluster = [{t.patient_id for t in members} for members in clusters.values()]
    assert {"low-1", "low-2"} in member_ids_per_cluster
    assert {"high-1", "high-2"} in member_ids_per_cluster


def test_cluster_twins_caps_n_clusters_at_sample_count(db):
    twin = _make_twin(db, "patient-1", 0.0, 0.0, 0.0)
    clusters = cluster_twins([twin], n_clusters=5)
    assert len(clusters) == 1


# ---------------------------------------------------------------------------
# vector_store.py
# ---------------------------------------------------------------------------


def test_upsert_and_query_similar_round_trips_through_chromadb(db, tmp_path):
    twin = _make_twin(db, "patient-1", 1.0, 2.0, 3.0)
    client = get_chroma_client(persist_directory=str(tmp_path / "chroma_db"))

    ids_by_domain = upsert_twin_embeddings(client, twin)
    assert set(ids_by_domain.keys()) == {"emr", "genomic", "wearable"}
    assert ids_by_domain["emr"] == embedding_id("patient-1", 1, "emr")

    result = query_similar(client, "emr", twin.emr_embedding, n_results=1)
    assert result["ids"][0][0] == ids_by_domain["emr"]


# ---------------------------------------------------------------------------
# reasoning.py -- API call mocked, no live key required
# ---------------------------------------------------------------------------


def test_generate_hypothesis_returns_valid_schema_matching_json(db):
    candidate_ids = [embedding_id("patient-1", 1, "genomic"), embedding_id("patient-2", 1, "wearable")]
    hypothesis_payload = {**VALID_HYPOTHESIS, "source_embedding_ids": candidate_ids}
    client = _mock_client_returning(hypothesis_payload)

    result = generate_hypothesis(client, "some structured cluster summary", candidate_ids)

    # Doesn't raise == valid against the strict, predefined schema.
    jsonschema.validate(instance=result, schema=HYPOTHESIS_JSON_SCHEMA)
    assert set(result.keys()) == {"subgroup_trait", "supporting_evidence", "confidence", "source_embedding_ids"}
    assert isinstance(result["subgroup_trait"], str) and result["subgroup_trait"]
    assert isinstance(result["supporting_evidence"], list) and result["supporting_evidence"]
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["source_embedding_ids"] == candidate_ids
    client.messages.create.assert_called_once()


def test_generate_hypothesis_rejects_schema_violations(db):
    candidate_ids = [embedding_id("patient-1", 1, "emr")]
    bad_payload = {**VALID_HYPOTHESIS, "confidence": 1.5, "source_embedding_ids": candidate_ids}
    client = _mock_client_returning(bad_payload)

    with pytest.raises(jsonschema.ValidationError):
        generate_hypothesis(client, "summary", candidate_ids)


def test_generate_hypothesis_rejects_hallucinated_source_ids(db):
    candidate_ids = [embedding_id("patient-1", 1, "emr")]
    payload = {**VALID_HYPOTHESIS, "source_embedding_ids": ["patient-999:v1:emr"]}
    client = _mock_client_returning(payload)

    with pytest.raises(ValueError):
        generate_hypothesis(client, "summary", candidate_ids)


def test_call_claude_retries_transient_errors_then_succeeds():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client = MagicMock(spec=anthropic.Anthropic)
    ok_response = SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=VALID_HYPOTHESIS)])
    client.messages.create.side_effect = [
        anthropic.APIConnectionError(request=request),
        anthropic.APIConnectionError(request=request),
        ok_response,
    ]

    result = _call_claude(client, model="claude-sonnet-5", max_tokens=10, messages=[])

    assert result is ok_response
    assert client.messages.create.call_count == 3


def test_call_claude_does_not_retry_non_transient_errors():
    client = MagicMock(spec=anthropic.Anthropic)
    client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com")), body=None
    )

    with pytest.raises(anthropic.AuthenticationError):
        _call_claude(client, model="claude-sonnet-5", max_tokens=10, messages=[])

    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Storage: every stored hypothesis is traceable to at least one source embedding
# ---------------------------------------------------------------------------


def test_store_hypothesis_persists_source_embedding_ids(db):
    candidate_ids = [embedding_id("patient-1", 1, "genomic"), embedding_id("patient-1", 1, "wearable")]
    hypothesis_payload = {**VALID_HYPOTHESIS, "source_embedding_ids": candidate_ids}

    row = store_hypothesis(db, hypothesis_payload, model="claude-sonnet-5")

    assert row.id is not None
    assert len(row.source_embedding_ids) >= 1
    assert set(row.source_embedding_ids) == set(candidate_ids)

    fetched = db.query(Hypothesis).filter_by(id=row.id).one()
    assert len(fetched.source_embedding_ids) >= 1
    assert fetched.subgroup_trait == VALID_HYPOTHESIS["subgroup_trait"]


def test_generate_and_store_hypothesis_end_to_end_with_mocked_api(db):
    twin = _make_twin(db, "patient-1", 1.0, 2.0, 3.0)
    candidate_ids = [embedding_id("patient-1", 1, "emr"), embedding_id("patient-1", 1, "genomic")]
    hypothesis_payload = {**VALID_HYPOTHESIS, "source_embedding_ids": candidate_ids}
    client = _mock_client_returning(hypothesis_payload)

    summary_text = format_patient_summary(twin)
    row = generate_and_store_hypothesis(db, client, summary_text, candidate_ids)

    assert row.id is not None
    assert len(row.source_embedding_ids) >= 1
    for eid in row.source_embedding_ids:
        assert eid in candidate_ids  # every reference is traceable to a real candidate embedding

    all_rows = db.query(Hypothesis).all()
    assert len(all_rows) == 1
    assert all(len(h.source_embedding_ids) >= 1 for h in all_rows)
