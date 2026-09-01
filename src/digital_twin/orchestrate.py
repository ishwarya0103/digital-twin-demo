"""Runs the three Layer 1 pipelines (architecture doc Section 3) and assembles their outputs
into a stored digital twin (Layer 2, `assembly.py`). This is orchestration only -- it invokes
each pipeline and hands their already-computed embeddings to `assemble_and_store_twin()`; it
does not itself compute, transform, or reason over anything.

Patient identity note: the EMR (Synthea FHIR), wearable (Exam Stress Dataset), and genomics
(this demo's VCF) source datasets are three independent public datasets with disjoint ID
spaces -- a Synthea Patient UUID, a wearable subject folder name ("S1".."S5"), and a VCF
sample column ("HG00096" etc.) never refer to the same real person. There is no natural
`patient_id` shared across all three. Rather than inventing one implicitly, every function
here takes each domain's *source* ID explicitly (`emr_patient_id`, `wearable_subject_id`,
`genomic_sample_id`) plus the `patient_id` the resulting twin should be stored under -- the
caller decides that correspondence, this module never guesses it.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from src.digital_twin.assembly import assemble_and_store_twin
from src.digital_twin.models import DigitalTwin
from src.emr_pipeline import PIPELINE_VERSION as EMR_PIPELINE_VERSION
from src.emr_pipeline import run_emr_pipeline_for_patient
from src.emr_pipeline.fhir_loader import discover_patient_bundles, load_patient_bundle
from src.genomics_pipeline import PIPELINE_VERSION as GENOMICS_PIPELINE_VERSION
from src.genomics_pipeline import run_genomics_pipeline
from src.genomics_pipeline.models import GenomicPathwayProfile
from src.wearable_pipeline import PIPELINE_VERSION as WEARABLE_PIPELINE_VERSION
from src.wearable_pipeline import run_wearable_pipeline
from src.wearable_pipeline.models import WearableProfile


def _find_emr_bundle_path(raw_dir: str, emr_patient_id: str, pipeline_version: str) -> Path:
    for path in discover_patient_bundles(raw_dir):
        if load_patient_bundle(path, pipeline_version).patient_id == emr_patient_id:
            return path
    raise ValueError(f"No EMR bundle under {raw_dir!r} has Patient.id == {emr_patient_id!r}")


def _find_profile(profiles: list, source_id: str, domain: str):
    for profile in profiles:
        if profile.patient_id == source_id:
            return profile
    raise ValueError(f"No {domain} profile with patient_id == {source_id!r} in pipeline output")


def orchestrate_twin_for_patient(
    db: Session,
    patient_id: str,
    emr_patient_id: str,
    wearable_subject_id: str,
    genomic_sample_id: str,
    emr_raw_dir: str = "data/raw/emr",
    emr_notes_dir: str = "data/raw/emr_notes",
    emr_pipeline_version: str = EMR_PIPELINE_VERSION,
    wearable_raw_dir: str = "data/raw/wearable",
    wearable_pipeline_version: str = WEARABLE_PIPELINE_VERSION,
    wearable_epochs: int = 150,
    wearable_seed: int = 0,
    genomics_raw_dir: str = "data/raw/genomics",
    genomics_pipeline_version: str = GENOMICS_PIPELINE_VERSION,
    wearable_profiles: list[WearableProfile] | None = None,
    genomic_profiles: list[GenomicPathwayProfile] | None = None,
) -> DigitalTwin:
    """Runs the EMR pipeline for `emr_patient_id`, the wearable pipeline for
    `wearable_subject_id`, and the genomics pipeline for `genomic_sample_id`, then stores the
    result as a new version of the digital twin identified by `patient_id`.

    The wearable and genomics pipelines are inherently cohort-level by design (one LSTM
    trained across all subjects; GWAS/PCA/pathway enrichment computed across the whole VCF's
    samples -- see Phase 2/3 notes in PROGRESS.md), so calling this function runs those two
    pipelines for the *whole* cohort under `wearable_raw_dir`/`genomics_raw_dir` even though
    only `wearable_subject_id`/`genomic_sample_id`'s result is used. To assemble twins for
    several patients without redundantly re-running those cohort pipelines each time, run them
    once yourself and pass the results in via `wearable_profiles`/`genomic_profiles` (or use
    `orchestrate_twins_for_cohort`, which does this automatically).
    """
    emr_bundle_path = _find_emr_bundle_path(emr_raw_dir, emr_patient_id, emr_pipeline_version)
    emr_result = run_emr_pipeline_for_patient(emr_bundle_path, emr_pipeline_version, notes_dir=emr_notes_dir)

    if wearable_profiles is None:
        wearable_profiles = run_wearable_pipeline(
            wearable_raw_dir, pipeline_version=wearable_pipeline_version, epochs=wearable_epochs, seed=wearable_seed
        )
    wearable_profile = _find_profile(wearable_profiles, wearable_subject_id, "wearable")

    if genomic_profiles is None:
        genomic_profiles = run_genomics_pipeline(genomics_raw_dir, pipeline_version=genomics_pipeline_version)
    genomic_profile = _find_profile(genomic_profiles, genomic_sample_id, "genomic")

    return assemble_and_store_twin(
        db,
        patient_id=patient_id,
        emr_embedding=emr_result["clinical_state_vector"],
        emr_pipeline_version=emr_result["pipeline_version"],
        genomic_embedding=genomic_profile.embedding,
        genomic_pipeline_version=genomic_profile.pipeline_version,
        wearable_embedding=wearable_profile.embedding,
        wearable_pipeline_version=wearable_profile.pipeline_version,
    )


def orchestrate_twins_for_cohort(
    db: Session,
    id_mappings: list[dict],
    emr_raw_dir: str = "data/raw/emr",
    emr_notes_dir: str = "data/raw/emr_notes",
    emr_pipeline_version: str = EMR_PIPELINE_VERSION,
    wearable_raw_dir: str = "data/raw/wearable",
    wearable_pipeline_version: str = WEARABLE_PIPELINE_VERSION,
    wearable_epochs: int = 150,
    wearable_seed: int = 0,
    genomics_raw_dir: str = "data/raw/genomics",
    genomics_pipeline_version: str = GENOMICS_PIPELINE_VERSION,
) -> list[DigitalTwin]:
    """Assembles a twin for each entry of `id_mappings`
    (`{"patient_id", "emr_patient_id", "wearable_subject_id", "genomic_sample_id"}` dicts,
    the caller's explicit choice of correspondence -- see module docstring), running the
    wearable and genomics cohort pipelines exactly once and reusing their output for every
    entry, rather than once per patient."""
    wearable_profiles = run_wearable_pipeline(
        wearable_raw_dir, pipeline_version=wearable_pipeline_version, epochs=wearable_epochs, seed=wearable_seed
    )
    genomic_profiles = run_genomics_pipeline(genomics_raw_dir, pipeline_version=genomics_pipeline_version)

    return [
        orchestrate_twin_for_patient(
            db,
            patient_id=mapping["patient_id"],
            emr_patient_id=mapping["emr_patient_id"],
            wearable_subject_id=mapping["wearable_subject_id"],
            genomic_sample_id=mapping["genomic_sample_id"],
            emr_raw_dir=emr_raw_dir,
            emr_notes_dir=emr_notes_dir,
            emr_pipeline_version=emr_pipeline_version,
            wearable_profiles=wearable_profiles,
            genomic_profiles=genomic_profiles,
        )
        for mapping in id_mappings
    ]
