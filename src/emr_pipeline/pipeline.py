from pathlib import Path

from sqlalchemy.orm import Session

from src.emr_pipeline.deidentify import deidentify_patient_record
from src.emr_pipeline.embedding import generate_clinical_state_vector
from src.emr_pipeline.fhir_loader import discover_patient_bundles, load_patient_bundle
from src.emr_pipeline.nlp_extraction import extract_events_from_notes
from src.emr_pipeline.summary import summarize_clinical_events
from src.emr_pipeline.timeline_builder import build_timeline
from src.governance.audit import log_audit_event

PIPELINE_VERSION = "emr-v0.1.0"


def run_emr_pipeline_for_patient(
    bundle_path: Path, pipeline_version: str = PIPELINE_VERSION, notes_dir=None, db: Session | None = None
) -> dict:
    raw_record = load_patient_bundle(bundle_path, pipeline_version, notes_dir=notes_dir)
    patient_id = raw_record.patient_id

    deid_record = deidentify_patient_record(raw_record, patient_id, pipeline_version)
    note_events = extract_events_from_notes(deid_record, patient_id, pipeline_version)
    timeline = build_timeline(deid_record, note_events, patient_id, pipeline_version)
    clinical_state_vector = generate_clinical_state_vector(timeline, patient_id, pipeline_version)
    clinical_summary = summarize_clinical_events(timeline)

    log_audit_event(
        pipeline_stage="emr_pipeline",
        action="run_emr_pipeline_for_patient",
        patient_id=patient_id,
        source_file=raw_record.source_file,
        pipeline_version=pipeline_version,
        db=db,
    )

    return {
        "patient_id": patient_id,
        "pipeline_version": pipeline_version,
        "demographics": deid_record.demographics,
        "notes": deid_record.notes,
        "timeline": timeline,
        "clinical_state_vector": clinical_state_vector,
        "clinical_summary": clinical_summary,
        "source_file": raw_record.source_file,
    }


def run_emr_pipeline(
    raw_dir: str = "data/raw/emr",
    pipeline_version: str = PIPELINE_VERSION,
    notes_dir: str = "data/raw/emr_notes",
    db: Session | None = None,
) -> list[dict]:
    return [
        run_emr_pipeline_for_patient(path, pipeline_version, notes_dir=notes_dir, db=db)
        for path in discover_patient_bundles(raw_dir)
    ]
