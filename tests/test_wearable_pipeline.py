import pytest

from src.wearable_pipeline.pipeline import run_wearable_pipeline

RAW_DIR = "data/raw/wearable"


@pytest.fixture(scope="module")
def profiles():
    return run_wearable_pipeline(raw_dir=RAW_DIR, epochs=150)


def test_five_subjects_processed(profiles):
    assert len(profiles) == 5
    assert {p.patient_id for p in profiles} == {"S1", "S2", "S3", "S4", "S5"}


@pytest.mark.parametrize("index", range(5))
def test_pipeline_runs_end_to_end_without_errors(profiles, index):
    profile = profiles[index]
    assert profile.patient_id
    assert profile.pipeline_version
    assert len(profile.windows) > 0
    assert profile.embedding.shape == (16,)


@pytest.mark.parametrize("index", range(5))
def test_activation_score_is_valid_probability(profiles, index):
    profile = profiles[index]
    for window in profile.windows:
        assert window.activation_score is not None
        assert 0.0 <= window.activation_score <= 1.0


@pytest.mark.parametrize("index", range(5))
def test_traceability_fields_present(profiles, index):
    profile = profiles[index]
    for window in profile.windows:
        assert window.patient_id == profile.patient_id
        assert window.pipeline_version == profile.pipeline_version
