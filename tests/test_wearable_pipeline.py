import pytest

from src.wearable_pipeline.pipeline import run_wearable_pipeline
from src.wearable_pipeline.summary import summarize_wearable_profile

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


# ---------------------------------------------------------------------------
# summary.py -- the actual activation score and a plain-language interpretation, not just the
# opaque 16-dim hidden-state embedding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(5))
def test_summarize_wearable_profile_matches_windows(profiles, index):
    profile = profiles[index]
    summary = summarize_wearable_profile(profile)

    assert set(summary.keys()) == {
        "mean_activation_score",
        "max_activation_score",
        "num_windows",
        "interpretation",
    }
    scores = [w.activation_score for w in profile.windows]
    assert summary["num_windows"] == len(scores)
    assert summary["mean_activation_score"] == pytest.approx(sum(scores) / len(scores))
    assert summary["max_activation_score"] == pytest.approx(max(scores))
    assert isinstance(summary["interpretation"], str) and summary["interpretation"]


@pytest.mark.parametrize(
    "mean_score,expected_interpretation",
    [(0.1, "low"), (0.5, "moderate"), (0.9, "high")],
)
def test_summarize_wearable_profile_interpretation_thresholds(mean_score, expected_interpretation):
    from src.wearable_pipeline.models import WearableProfile, WearableWindow

    window = WearableWindow(
        patient_id="p1",
        pipeline_version="v1",
        session="Final",
        window_type="onset",
        start_time=0.0,
        end_time=1.0,
        feature_sequence=None,
        activation_score=mean_score,
    )
    profile = WearableProfile(patient_id="p1", pipeline_version="v1", windows=[window], embedding=None)

    summary = summarize_wearable_profile(profile)
    assert expected_interpretation in summary["interpretation"]
