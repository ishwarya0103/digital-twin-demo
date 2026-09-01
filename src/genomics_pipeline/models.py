from dataclasses import dataclass

import numpy as np


@dataclass
class VariantAnnotation:
    """A variant's gene/functional context from SnpEff (stage 1) -- cohort-level, computed
    once from the shared input VCF rather than per patient."""

    pipeline_version: str
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alt: str
    gene: str | None
    consequence: str | None


@dataclass
class VariantAssociation:
    """A variant's GWAS association result from PLINK (stage 2) -- cohort-level."""

    pipeline_version: str
    chrom: str
    variant_id: str
    pos: int
    p_value: float | None


@dataclass
class PathwayEnrichment:
    """A pathway's cohort-level enrichment result from gseapy (stage 4): which of the
    architecture doc's pain-adjacent pathways (inflammation/nociception/drug metabolism) the
    cohort's annotated genes are over-represented in."""

    pipeline_version: str
    pathway: str
    genes: list[str]
    p_value: float
    combined_score: float


@dataclass
class GenomicPathwayProfile:
    """Stage 5 output: one patient's genomic pathway profile -- pathway-level genotype dosage
    (weighted by cohort enrichment), plus that patient's ancestry PCs from the population-
    stratification step, as one fixed-length structured feature vector."""

    patient_id: str
    pipeline_version: str
    pathway_names: list[str]  # fixed order, identical across all patients
    num_pcs: int
    embedding: np.ndarray  # (len(pathway_names) + num_pcs,)
