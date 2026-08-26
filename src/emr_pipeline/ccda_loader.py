"""Part of Stage 1 (Interoperability mapping): reads a patient's C-CDA "Summarization of
episode note" XML (Synthea's `output/ccda/` export) and flattens it into free text for the
notes pipeline.

Synthea's default FHIR export has no DocumentReference/notes at all -- structured data only.
Its CCDA export was tried as the source of real unstructured prose, but turned out to be a
structured document too: each section's narrative `<text>` is nearly always an HTML-style
table restating the same coded facts (Start/Stop/Description/Code columns), not
clinician-dictated sentences -- with occasional short prose asides (e.g. a Social History
section reading "There is no current social history except smoking status."). This loader
flattens those tables into one sentence per row ("Description: X, Code: Y.") so the free-text
stage still has genuine unstructured text -- XML narrative markup that must be parsed and
matched, not typed FHIR fields -- to run de-identification and clinical NLP extraction over,
while being honest that it's CCDA narrative text, not authored clinical notes.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from src.emr_pipeline.models import ClinicalNote

_NS = {"v3": "urn:hl7-org:v3"}


def _row_to_sentence(headers: list[str], cells: list[str]) -> str:
    parts = [f"{h}: {v}" for h, v in zip(headers, cells) if v and v.strip()]
    return (", ".join(parts) + ".") if parts else ""


def _flatten_section(section_el) -> list[str]:
    text_el = section_el.find("v3:text", _NS)
    if text_el is None:
        return []

    table = text_el.find("v3:table", _NS)
    if table is None:
        prose = " ".join(t.strip() for t in text_el.itertext() if t.strip())
        return [prose] if prose else []

    headers = [th.text or "" for th in table.findall(".//v3:thead/v3:tr/v3:th", _NS)]
    sentences = []
    for row in table.findall(".//v3:tbody/v3:tr", _NS):
        cells = [(td.text or "").strip() for td in row.findall("v3:td", _NS)]
        sentence = _row_to_sentence(headers, cells)
        if sentence:
            sentences.append(sentence)
    return sentences


def _document_date(root) -> str | None:
    effective_time = root.find("v3:effectiveTime", _NS)
    raw = effective_time.get("value") if effective_time is not None else None
    if not raw or len(raw) < 8:
        return None
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


def load_notes_from_ccda(xml_path: Path, patient_id: str, pipeline_version: str) -> list[ClinicalNote]:
    root = ET.parse(xml_path).getroot()

    sentences: list[str] = []
    for section in root.findall(".//v3:component/v3:section", _NS):
        sentences.extend(_flatten_section(section))

    if not sentences:
        return []

    return [
        ClinicalNote(
            patient_id=patient_id,
            pipeline_version=pipeline_version,
            date=_document_date(root),
            text=" ".join(sentences),
        )
    ]
