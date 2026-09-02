from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from src.genomics_pipeline.annotation import annotate_variants
from src.genomics_pipeline.config import PIPELINE_VERSION
from src.genomics_pipeline.embedding import build_patient_profile
from src.genomics_pipeline.gwas import (
    make_bed,
    parse_association_results,
    parse_pca_results,
    run_association_test,
    run_pca,
    write_synthetic_phenotype,
)
from src.genomics_pipeline.models import GenomicPathwayProfile
from src.genomics_pipeline.pathway_aggregation import run_pathway_enrichment
from src.genomics_pipeline.vcf_loader import discover_vcf, load_samples, load_variant_table
from src.governance.audit import log_audit_event


def run_genomics_pipeline(
    raw_dir: str = "data/raw/genomics",
    work_dir: str = "data/processed/genomics",
    pipeline_version: str = PIPELINE_VERSION,
    db: Session | None = None,
) -> list[GenomicPathwayProfile]:
    vcf_path = discover_vcf(raw_dir)
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(vcf_path)
    variant_table = load_variant_table(vcf_path)

    # Stage 1: variant annotation (SnpEff, standing in for ANNOVAR -- see config.py)
    annotations = annotate_variants(vcf_path, work_dir, pipeline_version)

    # Stage 2: GWAS (PLINK), against a synthetic phenotype -- see gwas.py
    bed_prefix = make_bed(vcf_path, work_dir)
    pheno_path = write_synthetic_phenotype(samples, work_dir)
    assoc_path = run_association_test(bed_prefix, pheno_path, work_dir)
    associations = parse_association_results(assoc_path, pipeline_version)
    variant_weights = {
        a.variant_id: -np.log10(max(a.p_value, 1e-300)) for a in associations if a.p_value is not None
    }

    # Stage 3: population stratification (PLINK --pca)
    eigenvec_path = run_pca(bed_prefix, work_dir)
    ancestry_by_sample = parse_pca_results(eigenvec_path)

    # Stage 4: pathway-level aggregation (gseapy)
    enrichments = run_pathway_enrichment(annotations, pipeline_version)

    # Stage 5: per-patient embedding
    profiles = [
        build_patient_profile(
            patient_id=sample,
            pipeline_version=pipeline_version,
            variant_table=variant_table,
            annotations=annotations,
            enrichments=enrichments,
            ancestry_pcs=ancestry_by_sample.get(sample, [0.0] * 2),
            variant_weights=variant_weights,
        )
        for sample in samples
    ]

    for sample in samples:
        log_audit_event(
            pipeline_stage="genomics_pipeline",
            action="run_genomics_pipeline",
            patient_id=sample,
            source_file=str(vcf_path),
            pipeline_version=pipeline_version,
            db=db,
        )

    return profiles
