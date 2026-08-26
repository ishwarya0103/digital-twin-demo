# Progress

## Current Phase

Phase 1 -- EMR pipeline

## Status

Complete

## Completed

**Phase 0 -- Project scaffolding**
- [x] Git repository initialized (`main` branch), local `user.name` / `user.email` configured
- [x] Folder structure created: `data/raw/`, `data/processed/`, `src/emr_pipeline/`,
      `src/wearable_pipeline/`, `src/genomics_pipeline/`, `src/digital_twin/`, `src/fusion_layer/`,
      `src/governance/`, `src/db/`, `api/`, `app/`, `tests/`, `docker/`
- [x] SQLAlchemy database layer (`src/db/session.py`, `src/db/base.py`) reading `DATABASE_URL` from
      the environment, defaulting to `sqlite:///data/processed/twin.db`; switching to PostgreSQL
      later requires only an env var change
- [x] `docker/Dockerfile` packaging the app; `docker-compose.yml` running it with `./data` mounted
      as a volume so the SQLite file persists outside the container
- [x] Commented-out `postgres` service block in `docker-compose.yml` as a migration placeholder
      (not active)
- [x] `chromadb` added to `requirements.txt` for later use as the embedding/vector store (not wired
      up yet)
- [x] `requirements.txt` covering current and future-phase dependencies
- [x] `tests/test_app_startup.py` -- confirms the app starts and connects to the database
- [x] `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`, `.dockerignore`

**Phase 1 -- EMR pipeline** (architecture doc Section 3.1), for 5 Synthea synthetic patients
- [x] `src/emr_pipeline/fhir_loader.py` -- Stage 1 (interoperability mapping). Reads Synthea FHIR
      R4 Bundle JSON from `data/raw/emr/`, normalizes Condition/MedicationRequest/Observation into
      `ClinicalEvent`s, decodes free-text `DocumentReference` notes (base64), and optionally merges
      narrative text parsed from a matching C-CDA XML in `data/raw/emr_notes/` (`ccda_loader.py`)
- [x] `src/emr_pipeline/deidentify.py` -- Stage 2 (de-identification), runs before anything
      downstream touches the data:
      - drops PHI-bearing structured fields (`name`, `birthDate`, `address`, `telecom`,
        `identifier`) and PHI-bearing extensions (`mothersMaidenName`, `birthPlace`) from the
        Patient resource
      - a deny-list scrubber redacts literal occurrences of the patient's own known identifiers in
        free text -- added after finding Synthea's digit-suffixed synthetic names (e.g. "Del587")
        aren't reliably caught by general-purpose NER
      - Presidio (`AnalyzerEngine` + `AnonymizerEngine`, `en_core_web_lg`) runs as a second, broader
        pass over the same text (PERSON, DATE_TIME, PHONE_NUMBER, US_SSN, LOCATION, etc.)
- [x] `src/emr_pipeline/nlp_extraction.py` -- Stage 3 (clinical NLP extraction). Pre-trained
      Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`, no fine-tuning). Candidate medication/
      diagnosis/symptom spans come from lexicon matching (medication and diagnosis terms drawn from
      the patient's own structured FHIR data, plus a generic symptom vocabulary); ClinicalBERT
      encodes the containing sentence for each match, so its forward pass does real work
      (contextual embeddings) rather than standing in for a token-classification head the base
      model doesn't have
- [x] `src/emr_pipeline/timeline_builder.py` -- Stage 4 (temporal alignment). Merges structured +
      note-derived events, sorts chronologically
- [x] `src/emr_pipeline/embedding.py` -- Stage 5 (embedding generation). Mean-pools per-event
      Bio_ClinicalBERT embeddings (Deep-Patient style) into a fixed 768-dim clinical state vector
      per patient
- [x] `src/emr_pipeline/pipeline.py` -- orchestrates all five stages; every function threads
      `patient_id` and `pipeline_version` through its inputs/outputs for traceability
- [x] `tests/test_emr_pipeline.py` -- for all 5 patients: non-empty/correctly-shaped clinical state
      vector, traceability fields present on every event, and no PHI fields remain after
      de-identification (checked against each patient's actual raw identifiers, not just a generic
      NER re-scan)
- [x] `data/raw/emr/` (5 Synthea FHIR bundles), `data/raw/emr_notes/` (matching C-CDA XML, same
      generation run), `data/raw/emr_metadata/` (Synthea's hospital/practitioner metadata, not
      patient records -- kept out of the pipeline's input directory)

## Test Status

Local venv (`.venv`, Python 3.12), not Docker -- see Known Issues:

```
17 passed, 1 warning in 81.15s (0:01:21)
```

`tests/test_app_startup.py::test_app_starts_and_connects_to_db` -- PASSED
`tests/test_emr_pipeline.py` (16 tests: 1 patient-count check + 3 per-patient checks x 5 patients)
-- PASSED

## Known Issues / Blockers

- Docker Desktop must be running before `docker compose build`/`run` -- confirm before invoking.
- Phase 1 tests ran via a local venv, not Docker. `docker/Dockerfile` now includes the
  `en_core_web_lg` spaCy download Presidio needs, but the image hasn't been rebuilt since -- do
  that (`docker compose build`) before running `tests/test_emr_pipeline.py` in a container; expect
  a slower first run since Bio_ClinicalBERT (~400MB) downloads from Hugging Face on first use
  inside the container rather than being pre-baked into the image.
- Synthea's default FHIR export has no free-text notes at all; even its C-CDA export is mostly
  structured tables restating coded data, not dictated prose -- except the `DocumentReference`
  resources in this regenerated FHIR set, which do contain genuine synthetic clinical notes
  (Chief Complaint / HPI / Assessment and Plan). The C-CDA merge is still wired in and adds some
  value/diversity but is largely redundant with those; could be dropped later to cut pipeline
  runtime if it matters.
- No pipeline logic yet for `src/wearable_pipeline/`, `src/genomics_pipeline/`,
  `src/digital_twin/`, `src/fusion_layer/`, `src/governance/` (still empty packages).
- `chromadb` is installed but not yet wired into any code path.
- Clinical state vectors are computed but not yet persisted to the database (`src/db/`) -- Phase 1
  only produces them in-memory via `run_emr_pipeline()`.

## Next Steps

- Phase 1: Wearable pipeline -- signal preprocessing, NeuroKit2 feature engineering, time-series
  model producing the physiological state profile ("ouch meter")
- Phase 1: Genomics pipeline -- variant annotation, GWAS, GSEA pathway aggregation producing the
  genomic pathway profile
- Persist EMR clinical state vectors via `src/db/` (and/or chromadb) so later phases can read them
  back rather than recomputing
- Phase 2: Digital Twin Abstraction Layer combining the three embeddings per patient
- Phase 3: Generative Semantic Fusion Layer (Anthropic API, structured/schema-constrained output)
