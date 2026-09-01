"""Stage 5 (embedding generation): a patient's genomic pathway profile -- a fixed-length
structured feature vector, no ML model needed (per the architecture doc: "Structured feature
vector (no ML model required at this stage -- statistical aggregation)").

Per-pathway score = sum of the patient's own genotype dosage (0/1/2 alt alleles) across
variants annotated to a gene in that pathway, weighted by how enriched that pathway is across
the whole cohort's annotated gene set (-log10 p-value from stage 4) -- so the profile reflects
both which variants a patient personally carries and which pathways those variants
collectively implicate for this cohort. Ancestry PCs (stage 3) are appended so population
stratification is represented directly in the same vector.
"""

import numpy as np
import pandas as pd

from src.genomics_pipeline.models import GenomicPathwayProfile, PathwayEnrichment, VariantAnnotation
from src.genomics_pipeline.pathway_aggregation import load_pathway_gene_sets


def build_patient_profile(
    patient_id: str,
    pipeline_version: str,
    variant_table: pd.DataFrame,
    annotations: list[VariantAnnotation],
    enrichments: list[PathwayEnrichment],
    ancestry_pcs: list[float],
    variant_weights: dict[str, float] | None = None,
) -> GenomicPathwayProfile:
    """variant_weights: {variant_id: -log10(GWAS p-value)}, from stage 2 -- lets a variant's
    cohort-level association strength scale how much it contributes to this patient's pathway
    scores, on top of how many alt alleles the patient actually carries. Defaults to 1.0 for
    any variant without an association result."""
    gene_by_variant = {a.variant_id: a.gene for a in annotations if a.gene}
    variant_weights = variant_weights or {}
    gene_sets = load_pathway_gene_sets()
    pathway_names = [e.pathway for e in enrichments]
    pathway_weight = {e.pathway: -np.log10(max(e.p_value, 1e-300)) for e in enrichments}

    raw_scores = dict.fromkeys(pathway_names, 0.0)
    for _, row in variant_table.iterrows():
        gene = gene_by_variant.get(row["ID"])
        if gene is None:
            continue
        dosage = row.get(patient_id)
        if dosage is None or (isinstance(dosage, float) and np.isnan(dosage)):
            continue
        assoc_weight = variant_weights.get(row["ID"], 1.0)
        for name in pathway_names:
            if gene in gene_sets.get(name, ()):
                raw_scores[name] += dosage * assoc_weight

    pathway_scores = [raw_scores[name] * pathway_weight[name] for name in pathway_names]
    embedding = np.array(pathway_scores + list(ancestry_pcs), dtype=np.float32)

    return GenomicPathwayProfile(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        pathway_names=pathway_names,
        num_pcs=len(ancestry_pcs),
        embedding=embedding,
    )
