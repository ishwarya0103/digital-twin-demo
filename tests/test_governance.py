from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.base import Base
from src.digital_twin.assembly import assemble_and_store_twin
from src.emr_pipeline.deidentify import known_identifier_strings
from src.emr_pipeline.fhir_loader import discover_patient_bundles, load_patient_bundle
from src.emr_pipeline.models import ClinicalNote
from src.emr_pipeline.pipeline import run_emr_pipeline_for_patient
from src.fusion_layer.formatting import embedding_id
from src.fusion_layer.reasoning import generate_and_store_hypothesis
from src.genomics_pipeline.pipeline import run_genomics_pipeline
from src.governance.fairness_check import check_subgroup_fairness
from src.governance.models import AuditLogEntry
from src.governance.phi_check import check_demographics_for_phi, check_notes_for_phi, check_patient_record_for_phi
from src.wearable_pipeline.pipeline import run_wearable_pipeline


@pytest.fixture
def db():
    """A fresh in-memory SQLite database per test -- isolated from the project's real
    data/processed/twin.db, and includes every table registered on Base (digital_twins,
    hypotheses, audit_log, ...) since they all share the same declarative Base."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _audit_rows(db, pipeline_stage: str, patient_id: str) -> list[AuditLogEntry]:
    return db.query(AuditLogEntry).filter_by(pipeline_stage=pipeline_stage, patient_id=patient_id).all()


def _mock_client_returning(hypothesis: dict) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    tool_use_block = SimpleNamespace(type="tool_use", input=hypothesis, id="toolu_fake", name="record_hypothesis")
    client.messages.create.return_value = SimpleNamespace(content=[tool_use_block])
    return client


# ---------------------------------------------------------------------------
# Every Phase 1-5 pipeline run produces a corresponding audit log entry
# ---------------------------------------------------------------------------


def test_emr_pipeline_run_produces_audit_log_entry(db):
    bundle_path = discover_patient_bundles("data/raw/emr")[0]
    result = run_emr_pipeline_for_patient(bundle_path, notes_dir="data/raw/emr_notes", db=db)

    rows = _audit_rows(db, "emr_pipeline", result["patient_id"])
    assert len(rows) == 1
    entry = rows[0]
    assert entry.action == "run_emr_pipeline_for_patient"
    assert entry.pipeline_version == result["pipeline_version"]
    assert entry.source_file == result["source_file"]
    assert entry.timestamp is not None


def test_wearable_pipeline_run_produces_audit_log_entry_per_subject(db):
    # epochs=5, not the pipeline's default 150 -- audit logging doesn't depend on how well the
    # model trained, only that the pipeline ran, so a fast run is enough to test it.
    profiles = run_wearable_pipeline(raw_dir="data/raw/wearable", epochs=5, db=db)
    assert profiles  # sanity: the pipeline actually processed subjects

    for profile in profiles:
        rows = _audit_rows(db, "wearable_pipeline", profile.patient_id)
        assert len(rows) == 1
        assert rows[0].action == "run_wearable_pipeline"
        assert rows[0].pipeline_version == profile.pipeline_version


def test_genomics_pipeline_run_produces_audit_log_entry_per_sample(db):
    profiles = run_genomics_pipeline(db=db)
    assert profiles

    for profile in profiles:
        rows = _audit_rows(db, "genomics_pipeline", profile.patient_id)
        assert len(rows) == 1
        assert rows[0].action == "run_genomics_pipeline"
        assert rows[0].pipeline_version == profile.pipeline_version


def test_digital_twin_assembly_produces_audit_log_entry(db):
    assemble_and_store_twin(
        db,
        patient_id="patient-1",
        emr_embedding=np.array([1.0, 2.0]),
        emr_pipeline_version="emr-v0.1.0",
        emr_summary={"diagnoses": [], "medications": [], "symptoms": []},
        genomic_embedding=np.array([3.0]),
        genomic_pipeline_version="genomics-v0.1.0",
        genomic_summary={"pathway_scores": {}, "ancestry_pcs": []},
        wearable_embedding=np.array([4.0, 5.0]),
        wearable_pipeline_version="wearable-v0.1.0",
        wearable_summary={"mean_activation_score": None, "max_activation_score": None, "num_windows": 0, "interpretation": "n/a"},
    )

    rows = _audit_rows(db, "digital_twin_assembly", "patient-1")
    assert len(rows) == 1
    assert rows[0].action == "assemble_and_store_twin"
    assert "emr=emr-v0.1.0" in rows[0].pipeline_version


def test_fusion_layer_hypothesis_generation_produces_audit_log_entry_per_patient(db):
    candidate_ids = [embedding_id("patient-1", 1, "emr"), embedding_id("patient-2", 1, "genomic")]
    hypothesis_payload = {
        "subgroup_trait": "test trait",
        "supporting_evidence": ["evidence 1"],
        "confidence": 0.5,
        "source_embedding_ids": candidate_ids,
    }
    client = _mock_client_returning(hypothesis_payload)

    generate_and_store_hypothesis(db, client, "cluster summary text", candidate_ids)

    for patient_id in ("patient-1", "patient-2"):
        rows = _audit_rows(db, "fusion_layer", patient_id)
        assert len(rows) == 1
        assert rows[0].action == "generate_and_store_hypothesis"


# ---------------------------------------------------------------------------
# PHI check: flags a deliberately-inserted PHI value, passes on clean data
# ---------------------------------------------------------------------------


def test_phi_check_flags_deliberately_inserted_phi_value():
    dirty_demographics = {"name": [{"given": ["Test"], "family": "Patient"}], "gender": "female"}
    demo_result = check_demographics_for_phi("patient-x", dirty_demographics)
    assert demo_result.passed is False
    assert any("name" in finding for finding in demo_result.findings)

    dirty_notes = [
        ClinicalNote(
            patient_id="patient-x",
            pipeline_version="test",
            date=None,
            text="Patient John Smith called on 555-123-4567 regarding a refill.",
        )
    ]
    notes_result = check_notes_for_phi("patient-x", dirty_notes)
    assert notes_result.passed is False
    assert notes_result.findings

    combined = check_patient_record_for_phi("patient-x", dirty_demographics, dirty_notes)
    assert combined.passed is False
    assert len(combined.findings) >= 2  # both the demographics leak and the notes leak


def test_phi_check_passes_on_clean_data():
    clean_demographics = {"gender": "female", "birthsex": "F"}  # no PHI_DEMOGRAPHIC_FIELDS keys
    demo_result = check_demographics_for_phi("patient-x", clean_demographics)
    assert demo_result.passed is True
    assert demo_result.findings == []

    clean_notes = [
        ClinicalNote(
            patient_id="patient-x",
            pipeline_version="test",
            date=None,
            text="Patient reports mild headache, prescribed ibuprofen 200mg as needed.",
        )
    ]
    notes_result = check_notes_for_phi("patient-x", clean_notes)
    assert notes_result.passed is True
    assert notes_result.findings == []

    combined = check_patient_record_for_phi("patient-x", clean_demographics, clean_notes)
    assert combined.passed is True
    assert combined.findings == []


def test_phi_check_on_real_deidentified_emr_output_finds_no_actual_patient_identifiers(db):
    """End-to-end: run the real EMR pipeline for one patient and confirm none of governance's
    PHI check findings correspond to this patient's *actual* known identifiers (name, phone,
    SSN, address, birth date -- Phase 1's own deny-list, see deidentify.known_identifier_strings).

    This deliberately does not assert zero findings: Presidio's re-scan of already-anonymized
    text can flag benign false positives that the anonymization pass itself introduced (e.g. a
    literal `http://snomed.info/sct` coding-system URL sometimes gets misclassified as PERSON
    once a neighboring `<US_SSN>`/`<DATE_TIME>` placeholder tag changes the surrounding
    context) -- that's the independent post-hoc check correctly re-verifying output, not a
    real PHI leak. What matters is that no flagged snippet is one of this patient's real
    identifiers.
    """
    bundle_path = discover_patient_bundles("data/raw/emr")[0]
    raw = load_patient_bundle(bundle_path, "test-raw", notes_dir="data/raw/emr_notes")
    result = run_emr_pipeline_for_patient(bundle_path, notes_dir="data/raw/emr_notes", db=db)

    check = check_patient_record_for_phi(result["patient_id"], result["demographics"], result["notes"])

    deny_terms = {term.lower() for term in known_identifier_strings(raw.demographics)}
    for finding in check.findings:
        assert not any(term in finding.lower() for term in deny_terms), (
            f"PHI check finding references a real patient identifier: {finding!r}"
        )


# ---------------------------------------------------------------------------
# Fairness check stub
# ---------------------------------------------------------------------------


def test_fairness_check_stub_runs_and_flags_small_sample():
    result = check_subgroup_fairness(
        y_pred=[0.8, 0.6, 0.3, 0.2, 0.5], sensitive_features=["A", "A", "B", "B", "B"]
    )
    assert set(result.metric_by_group.keys()) == {"A", "B"}
    assert result.n_per_group == {"A": 2, "B": 3}
    assert result.small_sample_warning is True  # 5 patients is always far below the real-world floor
