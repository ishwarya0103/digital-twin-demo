"""Turns a patient's genomic pathway profile (Stage 5's `GenomicPathwayProfile`) into a
labeled, human-readable summary -- named pathway scores and ancestry PCs -- for anything that
needs clinical language rather than the opaque, unlabeled feature vector. Reuses the profile's
own `pathway_names`; does not recompute anything.
"""

from src.genomics_pipeline.models import GenomicPathwayProfile


def summarize_pathway_profile(profile: GenomicPathwayProfile) -> dict:
    """{"pathway_scores": {pathway_name: score, ...}, "ancestry_pcs": [...]}. Pathway scores
    are the same values as the embedding's first len(pathway_names) dimensions, just labeled
    by name instead of left as anonymous vector positions; ancestry_pcs are the remaining
    num_pcs dimensions."""
    pathway_count = len(profile.pathway_names)
    scores = profile.embedding[:pathway_count]
    pcs = profile.embedding[pathway_count:]
    return {
        "pathway_scores": {name: float(score) for name, score in zip(profile.pathway_names, scores)},
        "ancestry_pcs": [float(pc) for pc in pcs],
    }
