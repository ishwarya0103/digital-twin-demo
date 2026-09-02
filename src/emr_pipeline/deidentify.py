"""Stage 2 (De-identification): strips PHI from a PatientRecord before anything downstream
(clinical NLP extraction, timeline building, embedding generation) touches it.

Three mechanisms, per the architecture doc's governance requirements ("Presidio, custom
NER-based scrubbers, tokenization service"):
  - Structured demographic fields (name, birthDate, address, telecom, identifier, and PHI-
    bearing extensions like mothersMaidenName/birthPlace) are dropped outright from the
    Patient resource -- they carry no clinical signal, only identity.
  - A deny-list pass redacts literal occurrences of the patient's own known identifiers
    (name, phone, SSN, address, birth date, pulled from the raw Patient resource before it's
    stripped) in free text. This exists because Synthea's synthetic given names carry a
    trailing digit suffix (e.g. "Del587"), which general-purpose NER does not reliably
    recognize as a PERSON -- exact-match deny-listing catches it where NER alone doesn't.
  - Presidio then runs its generic NER pass over the same text as a second, broader layer for
    anything the deny-list doesn't cover (other names mentioned, dates, phone-like patterns).
"""

import re
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.emr_pipeline.models import ClinicalNote, PatientRecord

PHI_DEMOGRAPHIC_FIELDS = ("name", "birthDate", "address", "telecom", "identifier")

# US Core patient extensions that carry identifying information rather than clinically
# useful demographics (race/ethnicity/birthsex are kept -- see paper section 6 on
# cross-cohort stratified bias evaluation).
PHI_EXTENSION_URLS = frozenset({
    "http://hl7.org/fhir/StructureDefinition/patient-mothersMaidenName",
    "http://hl7.org/fhir/StructureDefinition/patient-birthPlace",
})

PHI_ENTITIES = (
    "PERSON",
    "DATE_TIME",
    "PHONE_NUMBER",
    "US_SSN",
    "LOCATION",
    "EMAIL_ADDRESS",
    "MEDICAL_LICENSE",
)


@lru_cache(maxsize=1)
def get_presidio_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    return AnalyzerEngine(), AnonymizerEngine()


def scrub_demographics(demographics: dict) -> dict:
    scrubbed = {k: v for k, v in demographics.items() if k not in PHI_DEMOGRAPHIC_FIELDS}
    if "extension" in scrubbed:
        scrubbed["extension"] = [
            ext for ext in scrubbed["extension"] if ext.get("url") not in PHI_EXTENSION_URLS
        ]
    return scrubbed


def known_identifier_strings(demographics: dict) -> list[str]:
    """Literal identifying strings pulled from a patient's raw (pre-scrub) Patient resource --
    the deny-list used to catch identifiers a general-purpose NER model misses."""
    values: list[str] = []

    for name in demographics.get("name", []):
        values.extend(name.get("given", []))
        if name.get("family"):
            values.append(name["family"])

    for telecom in demographics.get("telecom", []):
        if telecom.get("value"):
            values.append(telecom["value"])

    for identifier in demographics.get("identifier", []):
        if identifier.get("value"):
            values.append(identifier["value"])

    for address in demographics.get("address", []):
        values.extend(address.get("line", []))
        if address.get("city"):
            values.append(address["city"])

    if demographics.get("birthDate"):
        values.append(demographics["birthDate"])

    return [v for v in values if v and len(v) > 2]


def _deny_list_scrub(text: str, deny_terms: list[str]) -> str:
    for term in sorted(deny_terms, key=len, reverse=True):
        text = re.sub(re.escape(term), "<PHI>", text, flags=re.IGNORECASE)
    return text


def scrub_note_text(text: str, deny_terms: list[str] | None = None) -> str:
    if deny_terms:
        text = _deny_list_scrub(text, deny_terms)
    analyzer, anonymizer = get_presidio_engines()
    results = analyzer.analyze(text=text, language="en", entities=list(PHI_ENTITIES))
    return anonymizer.anonymize(text=text, analyzer_results=results).text


def deidentify_patient_record(record: PatientRecord, patient_id: str, pipeline_version: str) -> PatientRecord:
    assert record.patient_id == patient_id

    deny_terms = known_identifier_strings(record.demographics)

    scrubbed_notes = [
        ClinicalNote(
            patient_id=patient_id,
            pipeline_version=pipeline_version,
            date=note.date,
            text=scrub_note_text(note.text, deny_terms),
        )
        for note in record.notes
    ]

    return PatientRecord(
        patient_id=patient_id,
        pipeline_version=pipeline_version,
        demographics=scrub_demographics(record.demographics),
        structured_events=record.structured_events,
        notes=scrubbed_notes,
        source_file=record.source_file,
    )
