import numpy as np
import pytest

from src.emr_pipeline.deidentify import (
    PHI_DEMOGRAPHIC_FIELDS,
    PHI_EXTENSION_URLS,
    known_identifier_strings,
)
from src.emr_pipeline.embedding import EMBEDDING_DIM
from src.emr_pipeline.fhir_loader import discover_patient_bundles, load_patient_bundle
from src.emr_pipeline.models import ClinicalEvent
from src.emr_pipeline.pipeline import run_emr_pipeline_for_patient
from src.emr_pipeline.summary import summarize_clinical_events

RAW_DIR = "data/raw/emr"
NOTES_DIR = "data/raw/emr_notes"


@pytest.fixture(scope="module")
def bundle_paths():
    paths = discover_patient_bundles(RAW_DIR)
    assert len(paths) == 5, f"expected 5 synthetic patients in {RAW_DIR}, found {len(paths)}"
    return paths


@pytest.fixture(scope="module")
def pipeline_results(bundle_paths):
    return [run_emr_pipeline_for_patient(p, notes_dir=NOTES_DIR) for p in bundle_paths]


@pytest.fixture(scope="module")
def raw_records(bundle_paths):
    # Loaded separately, pre-de-identification, purely so the de-id assertions below have
    # ground-truth identifiers to check against.
    return {path: load_patient_bundle(path, "test-raw", notes_dir=NOTES_DIR) for path in bundle_paths}


def test_five_patients_processed(pipeline_results):
    assert len(pipeline_results) == 5


@pytest.mark.parametrize("index", range(5))
def test_clinical_state_vector_nonempty_and_correctly_shaped(pipeline_results, index):
    vector = pipeline_results[index]["clinical_state_vector"]
    assert vector.shape == (EMBEDDING_DIM,)
    assert np.any(vector != 0)


@pytest.mark.parametrize("index", range(5))
def test_traceability_fields_present(pipeline_results, index):
    result = pipeline_results[index]
    assert result["patient_id"]
    assert result["pipeline_version"]
    for event in result["timeline"]:
        assert event.patient_id == result["patient_id"]
        assert event.pipeline_version == result["pipeline_version"]


@pytest.mark.parametrize("index", range(5))
def test_no_phi_fields_remain_after_deidentification(pipeline_results, bundle_paths, raw_records, index):
    result = pipeline_results[index]
    raw = raw_records[bundle_paths[index]]

    demographics = result["demographics"]
    for field in PHI_DEMOGRAPHIC_FIELDS:
        assert field not in demographics
    extension_urls = {ext.get("url") for ext in demographics.get("extension", [])}
    assert extension_urls.isdisjoint(PHI_EXTENSION_URLS)

    deny_terms = known_identifier_strings(raw.demographics)
    combined_note_text = " ".join(note.text.lower() for note in result["notes"])
    for term in deny_terms:
        assert term.lower() not in combined_note_text, (
            f"PHI value {term!r} leaked into notes for patient {result['patient_id']}"
        )


# ---------------------------------------------------------------------------
# summary.py -- labeled clinical summary (diagnoses/medications/symptoms), not the opaque
# clinical_state_vector
# ---------------------------------------------------------------------------


def test_summarize_clinical_events_groups_and_dedupes_by_type():
    timeline = [
        ClinicalEvent("p1", "v1", "diagnosis", "Essential hypertension (disorder)", source="structured"),
        ClinicalEvent("p1", "v1", "diagnosis", "Essential hypertension (disorder)", source="structured"),  # exact dup
        ClinicalEvent("p1", "v1", "medication", "Metformin 500 MG Oral Tablet", source="structured"),
        ClinicalEvent("p1", "v1", "medication", "metformin", source="note"),  # case-insensitive dup
        ClinicalEvent("p1", "v1", "symptom", "back pain", source="note"),
        ClinicalEvent("p1", "v1", "lab", "Glucose [Mass/volume] in Blood: 104.98 mg/dL", source="structured"),
    ]

    summary = summarize_clinical_events(timeline)

    assert summary == {
        "diagnoses": ["Essential hypertension (disorder)"],
        "medications": ["Metformin 500 MG Oral Tablet"],  # kept the first-seen casing
        "symptoms": ["back pain"],
    }
    assert "labs" not in summary  # too verbose/numeric to compress into a label list


def test_summarize_clinical_events_excludes_note_derived_diagnoses_and_medications():
    """Note-derived diagnosis/medication matches come from a "first significant word" heuristic
    that occasionally produces a generic non-clinical word (see summary.py's docstring) --
    only structured (FHIR-coded) diagnosis/medication text should be surfaced."""
    timeline = [
        ClinicalEvent("p1", "v1", "diagnosis", "index", source="note"),  # the noisy heuristic artifact
        ClinicalEvent("p1", "v1", "diagnosis", "Body mass index 30+ - obesity (finding)", source="structured"),
        ClinicalEvent("p1", "v1", "symptom", "headache", source="note"),  # symptoms are exempt from the filter
    ]

    summary = summarize_clinical_events(timeline)

    assert summary["diagnoses"] == ["Body mass index 30+ - obesity (finding)"]
    assert "index" not in summary["diagnoses"]
    assert summary["symptoms"] == ["headache"]


def test_run_emr_pipeline_for_patient_includes_labeled_clinical_summary(pipeline_results):
    """End-to-end, real pipeline output: clinical_summary sits alongside the opaque
    clinical_state_vector, and at least one real patient has non-trivial extracted content."""
    assert any(
        result["clinical_summary"]["diagnoses"] or result["clinical_summary"]["medications"]
        for result in pipeline_results
    )
    for result in pipeline_results:
        assert set(result["clinical_summary"].keys()) == {"diagnoses", "medications", "symptoms"}
