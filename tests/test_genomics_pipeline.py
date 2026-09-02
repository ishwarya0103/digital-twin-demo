import pytest

from src.genomics_pipeline.annotation import annotate_variants
from src.genomics_pipeline.config import PIPELINE_VERSION
from src.genomics_pipeline.gwas import (
    make_bed,
    parse_pca_results,
    run_association_test,
    run_pca,
    write_synthetic_phenotype,
)
from src.genomics_pipeline.pathway_aggregation import run_pathway_enrichment
from src.genomics_pipeline.pipeline import run_genomics_pipeline
from src.genomics_pipeline.summary import summarize_pathway_profile
from src.genomics_pipeline.vcf_loader import discover_vcf, load_samples

RAW_DIR = "data/raw/genomics"


@pytest.fixture(scope="module")
def vcf_path():
    return discover_vcf(RAW_DIR)


@pytest.fixture(scope="module")
def samples(vcf_path):
    found = load_samples(vcf_path)
    assert len(found) == 5, f"expected 5 samples in {vcf_path}, found {len(found)}"
    return found


@pytest.fixture(scope="module")
def plink_outputs(vcf_path, samples, tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("genomics_plink")
    bed_prefix = make_bed(vcf_path, work_dir)
    pheno_path = write_synthetic_phenotype(samples, work_dir)
    assoc_path = run_association_test(bed_prefix, pheno_path, work_dir)
    eigenvec_path = run_pca(bed_prefix, work_dir)
    return assoc_path, eigenvec_path


def test_plink_runs_without_errors(plink_outputs):
    assoc_path, eigenvec_path = plink_outputs
    assert assoc_path.exists()
    assert eigenvec_path.exists()


def test_plink_pca_covers_all_samples(plink_outputs, samples):
    _, eigenvec_path = plink_outputs
    ancestry = parse_pca_results(eigenvec_path)
    assert set(ancestry.keys()) == set(samples)


@pytest.fixture(scope="module")
def annotations(vcf_path, tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("genomics_annotate")
    return annotate_variants(vcf_path, work_dir, PIPELINE_VERSION)


@pytest.fixture(scope="module")
def enrichments(annotations):
    return run_pathway_enrichment(annotations, PIPELINE_VERSION)


def test_gseapy_runs_without_errors(enrichments):
    assert len(enrichments) > 0
    for enrichment in enrichments:
        assert 0.0 <= enrichment.p_value <= 1.0


@pytest.fixture(scope="module")
def profiles():
    return run_genomics_pipeline(raw_dir=RAW_DIR)


def test_five_samples_processed(profiles):
    assert len(profiles) == 5


@pytest.mark.parametrize("index", range(5))
def test_pathway_score_vector_expected_length(profiles, index):
    profile = profiles[index]
    expected_length = len(profile.pathway_names) + profile.num_pcs
    assert profile.embedding.shape == (expected_length,)


@pytest.mark.parametrize("index", range(5))
def test_traceability_fields_present(profiles, index):
    profile = profiles[index]
    assert profile.patient_id
    assert profile.pipeline_version


# ---------------------------------------------------------------------------
# summary.py -- named pathway scores, not just an unlabeled vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(5))
def test_summarize_pathway_profile_labels_every_pathway_by_name(profiles, index):
    profile = profiles[index]
    summary = summarize_pathway_profile(profile)

    assert set(summary.keys()) == {"pathway_scores", "ancestry_pcs"}
    # Every one of the profile's own named pathways appears as a key -- not left as an
    # anonymous vector position.
    assert set(summary["pathway_scores"].keys()) == set(profile.pathway_names)
    # The labeled scores are the same values as the embedding's own pathway dimensions, just
    # named instead of positional.
    for i, name in enumerate(profile.pathway_names):
        assert summary["pathway_scores"][name] == pytest.approx(float(profile.embedding[i]))
    assert len(summary["ancestry_pcs"]) == profile.num_pcs
