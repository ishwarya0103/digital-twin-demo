"""Governance check (architecture doc Section 7, "PHI separation | De-identification/
tokenization at ingestion, before features are built | Presidio, tokenization services"):
checks and reports whether any PHI-pattern fields remain in *processed* EMR data.

Reuses, rather than re-defines, what Phase 1's own de-identification (src/emr_pipeline/
deidentify.py) treats as PHI -- the same PHI_DEMOGRAPHIC_FIELDS/PHI_EXTENSION_URLS allow-list
for structured fields, and the same cached Presidio AnalyzerEngine + PHI_ENTITIES set for free
text -- so this check can never silently drift from what de-identification actually does.
It re-scans note text with Presidio fresh (not against the patient-specific deny-list Phase 1
also uses, which only catches identifiers already known ahead of time) -- this is a second,
independent check of the *output*, not a re-run of the de-identification logic itself.
"""

from dataclasses import dataclass, field

from src.emr_pipeline.deidentify import PHI_DEMOGRAPHIC_FIELDS, PHI_ENTITIES, PHI_EXTENSION_URLS, get_presidio_engines
from src.emr_pipeline.models import ClinicalNote


@dataclass
class PHICheckResult:
    patient_id: str
    passed: bool
    findings: list[str] = field(default_factory=list)


def check_demographics_for_phi(patient_id: str, demographics: dict) -> PHICheckResult:
    """Flags any PHI_DEMOGRAPHIC_FIELDS key or PHI_EXTENSION_URLS extension still present --
    these should have been dropped outright by `scrub_demographics()`, so any hit here means
    de-identification was skipped or a new PHI-bearing field was added without updating it."""
    findings = [
        f"demographics field {name!r} is present (should have been dropped)"
        for name in PHI_DEMOGRAPHIC_FIELDS
        if name in demographics
    ]

    extension_urls = {ext.get("url") for ext in demographics.get("extension", [])}
    findings.extend(f"PHI-bearing extension {url!r} is present" for url in extension_urls & PHI_EXTENSION_URLS)

    return PHICheckResult(patient_id=patient_id, passed=not findings, findings=findings)


def check_notes_for_phi(patient_id: str, notes: list[ClinicalNote]) -> PHICheckResult:
    """Runs Presidio's analyzer over already-processed note text, looking for the same entity
    types (PHI_ENTITIES) Phase 1's de-identification targets. A hit here means something
    Presidio-detectable survived de-identification."""
    analyzer, _ = get_presidio_engines()
    findings = []
    for note in notes:
        for result in analyzer.analyze(text=note.text, language="en", entities=list(PHI_ENTITIES)):
            snippet = note.text[result.start : result.end]
            findings.append(f"Presidio flagged {result.entity_type} ({snippet!r}) in a processed note")

    return PHICheckResult(patient_id=patient_id, passed=not findings, findings=findings)


def check_patient_record_for_phi(patient_id: str, demographics: dict, notes: list[ClinicalNote]) -> PHICheckResult:
    """The combined check: demographics fields/extensions plus a fresh Presidio scan of notes."""
    demo_result = check_demographics_for_phi(patient_id, demographics)
    notes_result = check_notes_for_phi(patient_id, notes)
    findings = demo_result.findings + notes_result.findings
    return PHICheckResult(patient_id=patient_id, passed=not findings, findings=findings)
