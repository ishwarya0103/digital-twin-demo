"""Stage 1 (Interoperability mapping): reads Synthea-format FHIR R4 Bundle JSON files and
normalizes each patient's structured fields (diagnoses, prescriptions, lab values) and
free-text notes into a PatientRecord.

Expected input layout:
    data/raw/emr/<anything>.json       -- one FHIR Bundle per patient, exactly as downloaded
                                           from Synthea's `output/fhir/` directory, filenames
                                           unchanged. Any resource types this loader doesn't
                                           recognize are ignored rather than raising, so real
                                           Synthea exports (hundreds of resources per patient)
                                           load without modification. Non-patient metadata
                                           files Synthea also emits (hospitalInformation*.json,
                                           practitionerInformation*.json) should be kept out of
                                           this directory (data/raw/emr_metadata/ is fine).
    data/raw/emr_notes/<same-stem>.xml -- optional, matching Synthea's `output/ccda/` export
                                           from the *same* generation run (so patient UUIDs
                                           line up). If present, its narrative text is added
                                           to that patient's notes. See ccda_loader.py for what
                                           "narrative" means for Synthea's CCDA output.
"""

import base64
import binascii
import json
from pathlib import Path

from src.emr_pipeline.ccda_loader import load_notes_from_ccda
from src.emr_pipeline.models import ClinicalEvent, ClinicalNote, PatientRecord

_SKIP_PREFIXES = ("hospitalInformation", "practitionerInformation")


def discover_patient_bundles(raw_dir) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(
        p
        for p in raw_dir.glob("*.json")
        if not p.name.startswith(_SKIP_PREFIXES)
    )


def _coding_text(codeable_concept: dict) -> tuple[str | None, str | None, str | None]:
    """Returns (display_text, code, code_system) from a FHIR CodeableConcept, preferring the
    top-level `text` field and falling back to the first coding entry."""
    if not codeable_concept:
        return None, None, None
    codings = codeable_concept.get("coding") or [{}]
    first = codings[0] if codings else {}
    text = codeable_concept.get("text") or first.get("display")
    return text, first.get("code"), first.get("system")


def _condition_to_event(resource: dict, patient_id: str, pipeline_version: str) -> ClinicalEvent | None:
    text, code, system = _coding_text(resource.get("code") or {})
    if not text:
        return None
    date = (
        resource.get("onsetDateTime")
        or (resource.get("onsetPeriod") or {}).get("start")
        or resource.get("recordedDate")
    )
    return ClinicalEvent(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        event_type="diagnosis",
        text=text,
        source="structured",
        date=date,
        code=code,
        code_system=system,
    )


def _medication_request_to_event(resource: dict, patient_id: str, pipeline_version: str) -> ClinicalEvent | None:
    text, code, system = _coding_text(resource.get("medicationCodeableConcept") or {})
    if not text:
        return None
    return ClinicalEvent(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        event_type="medication",
        text=text,
        source="structured",
        date=resource.get("authoredOn"),
        code=code,
        code_system=system,
    )


def _observation_to_event(resource: dict, patient_id: str, pipeline_version: str) -> ClinicalEvent | None:
    text, code, system = _coding_text(resource.get("code") or {})
    if not text:
        return None
    quantity = resource.get("valueQuantity")
    if quantity is not None:
        value = quantity.get("value")
        unit = quantity.get("unit", "")
        display_text = f"{text}: {value} {unit}".strip()
    else:
        display_text = text
    date = resource.get("effectiveDateTime") or (resource.get("effectivePeriod") or {}).get("start")
    return ClinicalEvent(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        event_type="lab",
        text=display_text,
        source="structured",
        date=date,
        code=code,
        code_system=system,
    )


def _document_reference_to_note(resource: dict, patient_id: str, pipeline_version: str) -> ClinicalNote | None:
    for content in resource.get("content", []):
        attachment = content.get("attachment", {})
        data = attachment.get("data")
        if not data:
            continue
        try:
            text = base64.b64decode(data).decode("utf-8", errors="replace")
        except (binascii.Error, ValueError):
            continue
        return ClinicalNote(
            patient_id=patient_id,
            pipeline_version=pipeline_version,
            date=resource.get("date"),
            text=text,
        )
    return None


_RESOURCE_HANDLERS = {
    "Condition": _condition_to_event,
    "MedicationRequest": _medication_request_to_event,
    "Observation": _observation_to_event,
}


def load_patient_bundle(path: Path, pipeline_version: str, notes_dir=None) -> PatientRecord:
    path = Path(path)
    bundle = json.loads(path.read_text())
    entries = bundle.get("entry", [])

    patient_resource = next(
        (e["resource"] for e in entries if e.get("resource", {}).get("resourceType") == "Patient"),
        None,
    )
    if patient_resource is None:
        raise ValueError(f"{path} does not contain a Patient resource")
    patient_id = patient_resource["id"]

    structured_events: list[ClinicalEvent] = []
    notes: list[ClinicalNote] = []

    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "DocumentReference":
            note = _document_reference_to_note(resource, patient_id, pipeline_version)
            if note is not None:
                notes.append(note)
            continue

        handler = _RESOURCE_HANDLERS.get(resource_type)
        if handler is None:
            continue
        event = handler(resource, patient_id, pipeline_version)
        if event is not None:
            structured_events.append(event)

    if notes_dir is not None:
        ccda_path = Path(notes_dir) / f"{path.stem}.xml"
        if ccda_path.exists():
            notes.extend(load_notes_from_ccda(ccda_path, patient_id, pipeline_version))

    return PatientRecord(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        demographics=patient_resource,
        structured_events=structured_events,
        notes=notes,
        source_file=str(path),
    )


def load_all_patients(raw_dir, pipeline_version: str, notes_dir=None) -> list[PatientRecord]:
    return [
        load_patient_bundle(p, pipeline_version, notes_dir=notes_dir)
        for p in discover_patient_bundles(raw_dir)
    ]
