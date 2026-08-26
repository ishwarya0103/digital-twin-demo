"""Stage 3 (Clinical NLP extraction): extracts medications, diagnoses, and symptom markers
from de-identified free-text notes, using a pre-trained Bio_ClinicalBERT (no fine-tuning).

Bio_ClinicalBERT ships without a token-classification head, so a randomly-initialized NER
head on top of it would produce noise, not extraction. Instead, candidate spans are found
with lexicon matching -- medication/diagnosis terms drawn from the patient's own structured
FHIR data, plus a generic symptom vocabulary -- and Bio_ClinicalBERT is used for what it's
actually pre-trained for: producing a contextual embedding of the sentence each match came
from. That embedding both travels with the event as evidence and feeds the embedding
generation stage (stage 5) directly, so ClinicalBERT's forward pass is doing real work, not
window dressing.
"""

import re
from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer

from src.emr_pipeline.models import ClinicalEvent, PatientRecord

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

SYMPTOM_LEXICON = [
    "shortness of breath", "blurred vision", "chest pain", "back pain", "joint pain",
    "abdominal pain", "headache", "fatigue", "dizziness", "nausea", "vomiting", "fever",
    "cough", "swelling", "numbness", "weakness", "congestion", "sneezing", "stiffness",
    "insomnia", "rash", "constipation", "diarrhea", "pain",
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z]+")


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def embed_text(text: str) -> list[float]:
    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0].mean(dim=0).tolist()


def _significant_terms(display_text: str) -> set[str]:
    """'Lisinopril 10 MG Oral Tablet' -> {'lisinopril 10 mg oral tablet', 'lisinopril'}."""
    lowered = display_text.lower()
    terms = {lowered}
    words = [w for w in _WORD_RE.findall(lowered) if len(w) > 4]
    if words:
        terms.add(words[0])
    return terms


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _drop_substring_matches(terms: set[str]) -> set[str]:
    """Keeps 'back pain' and drops the redundant generic 'pain' match in the same sentence."""
    return {t for t in terms if not any(t != other and t in other for other in terms)}


def extract_events_from_notes(
    record: PatientRecord, patient_id: str, pipeline_version: str
) -> list[ClinicalEvent]:
    assert record.patient_id == patient_id

    medication_terms: set[str] = set()
    diagnosis_terms: set[str] = set()
    for event in record.structured_events:
        if event.event_type == "medication":
            medication_terms |= _significant_terms(event.text)
        elif event.event_type == "diagnosis":
            diagnosis_terms |= _significant_terms(event.text)

    events: list[ClinicalEvent] = []
    seen: set[tuple[str, str, str | None]] = set()

    for note in record.notes:
        assert note.patient_id == patient_id
        for sentence in _sentences(note.text):
            lowered = sentence.lower()

            symptom_hits = _drop_substring_matches(
                {term for term in SYMPTOM_LEXICON if term in lowered}
            )
            medication_hits = {term for term in medication_terms if term in lowered}
            diagnosis_hits = {term for term in diagnosis_terms if term in lowered}

            sentence_matches = (
                [("medication", t) for t in medication_hits]
                + [("diagnosis", t) for t in diagnosis_hits]
                + [("symptom", t) for t in symptom_hits]
            )
            if not sentence_matches:
                continue

            sentence_embedding = embed_text(sentence)
            for event_type, term in sentence_matches:
                key = (event_type, term, note.date)
                if key in seen:
                    continue
                seen.add(key)
                events.append(
                    ClinicalEvent(
                        patient_id=patient_id,
                        pipeline_version=pipeline_version,
                        event_type=event_type,
                        text=term,
                        source="note",
                        date=note.date,
                        context_embedding=sentence_embedding,
                    )
                )

    return events
