"""Formats stored digital twins into structured text for the fusion layer's prompt-constrained
reasoning engine (architecture doc Section 5, "Layer 3"): "This layer performs cross-modal
reasoning but never touches raw EMR text, genomic sequences, or physiological signals directly
-- it consumes only the structured embeddings produced by Layer 1."

Built from each domain's *labeled clinical summary* (`DigitalTwin.{emr,genomic,wearable}_summary`
-- src/{emr,genomics,wearable}_pipeline/summary.py's output), not the raw embedding vectors: a
hypothesis reading "a subgroup characterized by specific autonomic activation patterns, poor
response to a class of analgesics, and enrichment in particular inflammatory pathways" (the
architecture doc's own Section 4.2 example) requires the reasoning engine to see medication
names, pathway names, and an activation interpretation -- not an embedding's L2 norm. The
summaries are still Layer 1's own already-extracted, already-structured output (diagnosis/
medication/symptom labels, named pathway scores, an activation-score interpretation), never raw
note text, variant sequences, or physiological signal data.
"""

from src.digital_twin.models import DigitalTwin
from src.digital_twin.retrieval import DOMAINS


def embedding_id(patient_id: str, version: int, domain: str) -> str:
    """Canonical ID for one domain's embedding of one twin version. Used consistently across
    chromadb entries (vector_store.py), formatted prompt text, and a hypothesis's
    source_embedding_ids, so all three can be cross-referenced against each other."""
    return f"{patient_id}:v{version}:{domain}"


def _format_emr_summary(summary: dict) -> str:
    parts = []
    for key, label in (("diagnoses", "diagnoses"), ("medications", "medications"), ("symptoms", "symptoms")):
        values = summary.get(key) or []
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    return "; ".join(parts) if parts else "no extracted diagnoses, medications, or symptoms"


def _format_genomic_summary(summary: dict) -> str:
    pathway_scores = summary.get("pathway_scores") or {}
    elevated = sorted(
        ((name, score) for name, score in pathway_scores.items() if score), key=lambda kv: -abs(kv[1])
    )
    if not elevated:
        return "no elevated genomic pathway signal"
    return "elevated pathways: " + ", ".join(f"{name} ({score:+.2f})" for name, score in elevated)


def _format_wearable_summary(summary: dict) -> str:
    interpretation = summary.get("interpretation", "no wearable activation data available")
    mean_score = summary.get("mean_activation_score")
    if mean_score is None:
        return interpretation
    num_windows = summary.get("num_windows", 0)
    return f"{interpretation} (mean activation score {mean_score:.2f} across {num_windows} windows)"


_DOMAIN_FORMATTERS = {
    "emr": _format_emr_summary,
    "genomic": _format_genomic_summary,
    "wearable": _format_wearable_summary,
}


def _domain_summary_line(twin: DigitalTwin, domain: str) -> str:
    summary = getattr(twin, f"{domain}_summary") or {}
    pipeline_version = getattr(twin, f"{domain}_pipeline_version")
    eid = embedding_id(twin.patient_id, twin.version, domain)
    detail = _DOMAIN_FORMATTERS[domain](summary)
    return f"  - {domain} (embedding_id={eid}, pipeline_version={pipeline_version}): {detail}"


def format_patient_summary(twin: DigitalTwin) -> str:
    """Structured clinical-language text summarizing one patient's twin: one labeled line per
    domain, built from that domain's labeled summary -- never the embeddings' own raw values or
    any domain's raw source data."""
    lines = [f"Patient {twin.patient_id} (twin version {twin.version}):"]
    lines.extend(_domain_summary_line(twin, domain) for domain in DOMAINS)
    return "\n".join(lines)


def _shared_emr_terms(twins: list[DigitalTwin]) -> str:
    diagnoses = [set((t.emr_summary or {}).get("diagnoses") or []) for t in twins]
    medications = [set((t.emr_summary or {}).get("medications") or []) for t in twins]
    symptoms = [set((t.emr_summary or {}).get("symptoms") or []) for t in twins]

    shared_diagnoses = set.intersection(*diagnoses) if diagnoses else set()
    shared_medications = set.intersection(*medications) if medications else set()
    shared_symptoms = set.intersection(*symptoms) if symptoms else set()

    parts = []
    if shared_diagnoses:
        parts.append(f"diagnoses shared by every member: {', '.join(sorted(shared_diagnoses))}")
    if shared_medications:
        parts.append(f"medications shared by every member: {', '.join(sorted(shared_medications))}")
    if shared_symptoms:
        parts.append(f"symptoms shared by every member: {', '.join(sorted(shared_symptoms))}")
    return "; ".join(parts) if parts else "no single diagnosis, medication, or symptom shared by every member"


def _shared_genomic_pathways(twins: list[DigitalTwin]) -> str:
    per_patient_elevated = [
        {name for name, score in ((t.genomic_summary or {}).get("pathway_scores") or {}).items() if score}
        for t in twins
    ]
    shared = set.intersection(*per_patient_elevated) if per_patient_elevated else set()
    if not shared:
        return "no single pathway elevated in every member"
    return "pathways elevated in every member: " + ", ".join(sorted(shared))


def _wearable_range(twins: list[DigitalTwin]) -> str:
    scores = [
        (t.wearable_summary or {}).get("mean_activation_score")
        for t in twins
        if (t.wearable_summary or {}).get("mean_activation_score") is not None
    ]
    if not scores:
        return "no wearable activation data available across this cluster"
    interpretations = sorted({(t.wearable_summary or {}).get("interpretation") for t in twins if t.wearable_summary})
    return (
        f"activation scores range {min(scores):.2f}-{max(scores):.2f} across the cluster "
        f"(interpretations present: {', '.join(i for i in interpretations if i)})"
    )


def format_cluster_summary(twins: list[DigitalTwin], cluster_id: int | str) -> str:
    """Structured clinical-language text summarizing a cluster of patients grouped by embedding
    similarity (clustering.py): what's shared across the whole cluster per domain, followed by
    each member's own per-domain summary -- the reasoning engine sees both the group-level
    signal and which patients contribute to it."""
    if not twins:
        raise ValueError("Cannot format a cluster summary for an empty patient list")

    lines = [f"Cluster {cluster_id} ({len(twins)} patients: {', '.join(t.patient_id for t in twins)}):"]
    lines.append(f"  emr: {_shared_emr_terms(twins)}")
    lines.append(f"  genomic: {_shared_genomic_pathways(twins)}")
    lines.append(f"  wearable: {_wearable_range(twins)}")

    lines.append("")
    lines.append("Per-patient detail:")
    for twin in twins:
        lines.append(format_patient_summary(twin))

    return "\n".join(lines)
