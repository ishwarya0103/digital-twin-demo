import numpy as np
import pytest

from src.emr_pipeline.deidentify import (
    PHI_DEMOGRAPHIC_FIELDS,
    PHI_EXTENSION_URLS,
    known_identifier_strings,
)
from src.emr_pipeline.embedding import EMBEDDING_DIM
from src.emr_pipeline.fhir_loader import discover_patient_bundles, load_patient_bundle
from src.emr_pipeline.pipeline import run_emr_pipeline_for_patient

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
