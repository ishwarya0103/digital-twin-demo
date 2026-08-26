from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClinicalEvent:
    """A single dated fact about a patient, traceable to its source pipeline run."""

    patient_id: str
    pipeline_version: str
    event_type: str  # "diagnosis" | "medication" | "lab" | "symptom"
    text: str
    source: str  # "structured" | "note"
    date: Optional[str] = None
    code: Optional[str] = None
    code_system: Optional[str] = None
    # Contextual embedding from the note sentence a note-derived event was found in
    # (populated in the NLP extraction stage). None for structured (FHIR-coded) events --
    # the embedding-generation stage computes one on demand for those instead.
    context_embedding: Optional[list[float]] = None


@dataclass
class ClinicalNote:
    patient_id: str
    pipeline_version: str
    date: Optional[str]
    text: str


@dataclass
class PatientRecord:
    """Output of the FHIR loader: one patient's structured events and free-text notes,
    still containing raw demographic PHI until it passes through de-identification.
    """

    patient_id: str
    pipeline_version: str
    demographics: dict
    structured_events: list[ClinicalEvent]
    notes: list[ClinicalNote]
    source_file: str
