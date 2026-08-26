"""Stage 4 (Temporal alignment): merges structured FHIR events with note-derived events and
sorts them into one chronological timeline per patient. Undated events (rare -- e.g. a
medication with no authoredOn) sort last rather than being dropped, since they still carry
evidence for the embedding-generation stage.
"""

from src.emr_pipeline.models import ClinicalEvent, PatientRecord


def build_timeline(
    record: PatientRecord,
    note_events: list[ClinicalEvent],
    patient_id: str,
    pipeline_version: str,
) -> list[ClinicalEvent]:
    assert record.patient_id == patient_id

    all_events = list(record.structured_events) + list(note_events)
    for event in all_events:
        assert event.patient_id == patient_id
        assert event.pipeline_version == pipeline_version

    return sorted(all_events, key=lambda e: (e.date is None, e.date or ""))
