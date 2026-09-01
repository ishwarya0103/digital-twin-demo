"""Stage 4 (pathway-level aggregation): collapses variant-level gene annotations into
pathway-level enrichment scores via gseapy, against a small KEGG-derived gene-set collection
curated for relevance to the paper's chronic-pain use case -- inflammation, nociception, drug
metabolism (see pain_pathways.gmt, fetched once from Enrichr's KEGG_2021_Human library).

Local `gp.enrich()` (hypergeometric over-representation test) is used rather than
`gp.enrichr()`/`gp.prerank()`: this cohort has too few annotated genes for permutation-based
ranked GSEA to be stable, and enrich() needs no network access at pipeline-run time -- only
pain_pathways.gmt's original fetch did.
"""

import gseapy as gp

from src.genomics_pipeline.config import PATHWAY_GMT_PATH
from src.genomics_pipeline.models import PathwayEnrichment, VariantAnnotation


def load_pathway_gene_sets(gmt_path=PATHWAY_GMT_PATH) -> dict[str, list[str]]:
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            gene_sets[parts[0]] = parts[2:]
    return gene_sets


def run_pathway_enrichment(
    annotations: list[VariantAnnotation], pipeline_version: str, gmt_path=PATHWAY_GMT_PATH
) -> list[PathwayEnrichment]:
    gene_sets = load_pathway_gene_sets(gmt_path)
    genes_hit = sorted({a.gene for a in annotations if a.gene})

    def _neutral(name: str) -> PathwayEnrichment:
        return PathwayEnrichment(
            pipeline_version=pipeline_version, pathway=name, genes=[], p_value=1.0, combined_score=0.0
        )

    if not genes_hit:
        return [_neutral(name) for name in gene_sets]

    background = sorted({g for genes in gene_sets.values() for g in genes})
    result = gp.enrich(gene_list=genes_hit, gene_sets=gene_sets, background=background, outdir=None)
    df = result.results
    # gseapy returns [] instead of an empty DataFrame when none of the hit genes overlap any
    # pathway gene set at all (as opposed to overlapping some but not others).
    if isinstance(df, list):
        return [_neutral(name) for name in gene_sets]

    enrichments = []
    for name in gene_sets:
        row = df[df["Term"] == name]
        if row.empty:
            enrichments.append(_neutral(name))
            continue
        row = row.iloc[0]
        overlap_genes = row["Genes"].split(";") if isinstance(row["Genes"], str) and row["Genes"] else []
        enrichments.append(
            PathwayEnrichment(
                pipeline_version=pipeline_version,
                pathway=name,
                genes=overlap_genes,
                p_value=float(row["P-value"]),
                combined_score=float(row["Combined Score"]),
            )
        )
    return enrichments
