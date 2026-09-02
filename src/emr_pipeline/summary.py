"""Turns a patient's extracted clinical timeline (Stage 3/4's `ClinicalEvent`s) into a labeled,
human-readable summary -- deduplicated diagnosis/medication/symptom names -- for anything that
needs clinical language rather than the opaque 768-dim clinical state vector (Stage 5). Reuses
Stage 3/4's own already-extracted output; does not re-extract, re-interpret, or touch any note
text itself.

Deliberately excludes lab events: `Observation`-derived event text bakes in the specific
measured value for each individual draw ("Glucose [Mass/volume] in Blood: 104.98 mg/dL"), so
distinct lab text does not compress into a short label list the way diagnosis/medication names
do -- a patient with a few years of routine labs produces hundreds of near-duplicate, distinctly
worded entries. Diagnoses/medications and symptoms are closed-ish vocabularies (a condition or
drug is named consistently across mentions); lab *readings* are not, so listing them here would
just reproduce a wall of numbers under a different label.

Diagnosis/medication events only count here when `source == "structured"` (straight from FHIR
Condition/MedicationRequest text). Note-derived diagnosis/medication matches
(`nlp_extraction.py`) are found via a "first word over 4 characters" heuristic applied to
already-structured text, which occasionally produces a generic, non-clinical word (e.g. "index"
out of "Body mass index 30+ - obesity (finding)", or "history" out of "history of coronary
artery bypass grafting") -- harmless when diluted into a 768-dim embedding mean-pool, but wrong
to surface as if it were a diagnosis in a human/LLM-facing summary. Symptom matches are exempt
from this filter: `SYMPTOM_LEXICON` is a small, curated, always-clean vocabulary regardless of
which source text they were matched in, not derived via that heuristic.
"""

from src.emr_pipeline.models import ClinicalEvent

_STRUCTURED_ONLY_EVENT_TYPES = {"diagnosis", "medication"}
_BUCKET_BY_EVENT_TYPE = {
    "diagnosis": "diagnoses",
    "medication": "medications",
    "symptom": "symptoms",
}


def summarize_clinical_events(timeline: list[ClinicalEvent]) -> dict:
    """{"diagnoses": [...], "medications": [...], "symptoms": [...]}, each a sorted list of
    that event_type's distinct `text` values. Deduplicates case-insensitively (structured FHIR
    text and note-derived lexicon matches often differ only in case/capitalization, e.g.
    "Lisinopril 10 MG Oral Tablet" vs. "lisinopril") but keeps whichever casing was seen first,
    rather than forcing a single style."""
    buckets: dict[str, dict[str, str]] = {bucket: {} for bucket in _BUCKET_BY_EVENT_TYPE.values()}

    for event in timeline:
        bucket = _BUCKET_BY_EVENT_TYPE.get(event.event_type)
        if bucket is None:
            continue
        if event.event_type in _STRUCTURED_ONLY_EVENT_TYPES and event.source != "structured":
            continue
        text = event.text.strip()
        if not text:
            continue
        buckets[bucket].setdefault(text.lower(), text)

    return {bucket: sorted(values.values()) for bucket, values in buckets.items()}
