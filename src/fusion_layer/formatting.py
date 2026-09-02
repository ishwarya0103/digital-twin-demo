"""Formats stored digital twins into structured text for the fusion layer's prompt-constrained
reasoning engine (architecture doc Section 5, "Layer 3"): "This layer performs cross-modal
reasoning but never touches raw EMR text, genomic sequences, or physiological signals directly
-- it consumes only the structured embeddings produced by Layer 1."

DigitalTwin rows (Layer 2) never contain raw domain data to begin with -- only the
already-abstracted embedding vectors -- so the risk this module guards against is different:
dumping hundreds of raw embedding floats into a prompt is neither meaningfully "structured
text" nor something the reasoning engine can interpret. Instead, each domain contributes
labeled summary statistics (dimensionality, L2 norm, mean, std) plus its traceability
metadata (pipeline_version, embedding_id) -- interpretable, and still nothing beyond what
Layer 1/2 already produced.
"""

import numpy as np

from src.digital_twin.models import DigitalTwin
from src.digital_twin.retrieval import DOMAINS


def embedding_id(patient_id: str, version: int, domain: str) -> str:
    """Canonical ID for one domain's embedding of one twin version. Used consistently across
    chromadb entries (vector_store.py), formatted prompt text, and a hypothesis's
    source_embedding_ids, so all three can be cross-referenced against each other."""
    return f"{patient_id}:v{version}:{domain}"


def _domain_summary_line(twin: DigitalTwin, domain: str) -> str:
    embedding = np.asarray(getattr(twin, f"{domain}_embedding"), dtype=np.float64)
    pipeline_version = getattr(twin, f"{domain}_pipeline_version")
    eid = embedding_id(twin.patient_id, twin.version, domain)
    return (
        f"  - {domain}: embedding_id={eid}, pipeline_version={pipeline_version}, "
        f"dim={embedding.shape[0]}, L2_norm={np.linalg.norm(embedding):.4f}, "
        f"mean={embedding.mean():.4f}, std={embedding.std():.4f}"
    )


def format_patient_summary(twin: DigitalTwin) -> str:
    """Structured text summarizing one patient's twin: one labeled line per domain, summary
    statistics only -- never the embeddings' own raw values or any domain's raw source data."""
    lines = [f"Patient {twin.patient_id} (twin version {twin.version}):"]
    lines.extend(_domain_summary_line(twin, domain) for domain in DOMAINS)
    return "\n".join(lines)


def format_cluster_summary(twins: list[DigitalTwin], cluster_id: int | str) -> str:
    """Structured text summarizing a cluster of patients grouped by embedding similarity
    (clustering.py): cluster-level aggregate statistics per domain, followed by each member's
    own per-domain summary -- the reasoning engine sees both the group-level signal and which
    patients (and which of their embeddings) contribute to it."""
    if not twins:
        raise ValueError("Cannot format a cluster summary for an empty patient list")

    lines = [f"Cluster {cluster_id} ({len(twins)} patients: {', '.join(t.patient_id for t in twins)}):"]
    for domain in DOMAINS:
        embeddings = np.asarray([getattr(t, f"{domain}_embedding") for t in twins], dtype=np.float64)
        pipeline_versions = sorted({getattr(t, f"{domain}_pipeline_version") for t in twins})
        lines.append(
            f"  {domain} domain (pipeline_version(s)={pipeline_versions}): "
            f"mean L2_norm across cluster={np.linalg.norm(embeddings, axis=1).mean():.4f}, "
            f"mean per-dimension value={embeddings.mean():.4f}, "
            f"mean per-dimension std across cluster={embeddings.std(axis=0).mean():.4f}"
        )

    lines.append("")
    lines.append("Per-patient detail:")
    for twin in twins:
        lines.append(format_patient_summary(twin))

    return "\n".join(lines)
