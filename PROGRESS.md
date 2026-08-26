# Progress

## Current Phase

Phase 2 -- Wearable pipeline

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

**Phase 2 -- Wearable pipeline** (architecture doc Section 3.2), for 5 subjects (S1-S5) of the
Wearable Exam Stress Dataset (Empatica E4 exports, 3 sessions each: Final/Midterm 1/Midterm 2)
- [x] `src/wearable_pipeline/signal_loader.py` -- reads each session's HR/EDA/BVP/TEMP channels
      (each with its own start time -- Empatica's HR channel starts ~10s after the others) and
      tags.csv event markers. IBI.csv (Empatica's onboard beat detector) is read but not used for
      HRV (see feature_extraction.py)
- [x] `src/wearable_pipeline/preprocessing.py` -- Stage 1. Per-channel Butterworth filtering
      (SciPy) with physiologically appropriate cutoffs (BVP bandpass 0.5-8Hz, EDA/TEMP/HR lowpass);
      normalization is applied later to extracted *feature* vectors rather than raw signal, so
      NeuroKit2's physically-calibrated processing downstream still sees real units
- [x] `src/wearable_pipeline/segmentation.py` -- Stage 2. Windows are centered on tags.csv
      timestamps where present (mostly "Midterm 1" sessions); where a session has no tags (most
      "Final" sessions), the session's own start/end stand in as the onset/resolution boundary
- [x] `src/wearable_pipeline/feature_extraction.py` -- Stage 3. NeuroKit2 HRV (MeanNN/SDNN/RMSSD)
      via `ppg_process`/`hrv_time` run directly on raw BVP -- not on IBI.csv, which turned out too
      sparse (median ~2s gaps between detected beats, with multi-minute dropouts) for reliable
      interval statistics -- plus EDA tonic/phasic/SCR-count (`eda_process`) and simple temp/HR
      stats, one feature vector per 60s sub-interval of each window
- [x] `src/wearable_pipeline/model.py` -- Stage 4. A small single-layer PyTorch LSTM (16 hidden
      units) mapping a window's feature sequence to a sigmoid activation score ("ouch meter").
      This dataset has no real pain/symptom self-reports to train against (it's exam stress, not
      pain), so the training target is a heuristic composite autonomic-activation index (elevated
      HR, suppressed HRV, elevated EDA, z-scored across the cohort) -- the LSTM genuinely learns to
      approximate that target from the raw sequence rather than just recomputing it. One model is
      trained across the whole 5-subject cohort, not one per patient, per the architecture doc's
      "same time-series model, trained against patient-reported labels" framing
- [x] `src/wearable_pipeline/embedding.py` -- Stage 5. Mean-pools each patient's per-window LSTM
      hidden states into a fixed 16-dim wearable-derived physiological state profile, alongside the
      ordered list of windows (the longitudinal ouch-meter trajectory)
- [x] `src/wearable_pipeline/pipeline.py` -- orchestrates all five stages across all subjects/
      sessions; `patient_id`/`pipeline_version` threaded through every `WearableWindow`/
      `WearableProfile`, same convention as Phase 1
- [x] `tests/test_wearable_pipeline.py` -- for all 5 subjects: pipeline runs end-to-end without
      errors, every window's activation score is a valid probability in [0, 1], traceability
      fields present throughout
- [x] `data/raw/wearable/` populated locally with the 5-subject dataset, but -- unlike
      `data/raw/emr/` -- gitignored rather than committed: ~165MB of raw CSVs (23.7M lines) was a
      big enough jump from Phase 1's ~16MB that it warranted asking rather than assuming the same
      precedent applied. See README.md's "Wearable data" section for how to re-populate it

## Test Status

Verified in both environments:

Local venv (`.venv`, Python 3.12):
```
17 passed, 1 warning in 81.15s (0:01:21)
```

Docker (`docker compose build` then `docker compose run --rm app pytest -v`, 2026-08-25), image now
includes the `en_core_web_lg` spaCy download and a Bio_ClinicalBERT pre-fetch, both baked in at
build time so neither re-downloads on container run:
```
17 passed, 1 warning in 244.18s (0:04:04)
```
Only modestly faster than the pre-caching run (299.81s) -- the ~400MB Hugging Face download
accounted for some of the difference, but most of the runtime is genuinely CPU-bound
Bio_ClinicalBERT inference across the ~1,000+ note-derived events per patient (see Known Issues),
not network time. The caching change still did its job: containers no longer depend on network
access to Hugging Face at all once built.

`tests/test_app_startup.py::test_app_starts_and_connects_to_db` -- PASSED
`tests/test_emr_pipeline.py` (16 tests: 1 patient-count check + 3 per-patient checks x 5 patients)
-- PASSED

Local venv, full suite including the new wearable tests (2026-08-26):
```
33 passed, 1 warning in 90.35s (0:01:30)
```
`tests/test_wearable_pipeline.py` (16 tests: 1 subject-count check + 3 per-subject checks x 5
subjects) -- PASSED. Not yet re-verified inside Docker (see Known Issues).

## Known Issues / Blockers

- Docker Desktop must be running before `docker compose build`/`run` -- confirm before invoking.
- Most of the ~4 minute Docker test runtime is CPU-bound Bio_ClinicalBERT inference (one forward
  pass per matched medication/diagnosis/symptom mention -- ~1,000+ per patient for the richer
  synthetic histories), not model loading or download. If this becomes a hot path, the cheaper fix
  is cutting down redundant note-derived matches (e.g. dropping the C-CDA merge, see below) rather
  than infrastructure changes.
- Synthea's default FHIR export has no free-text notes at all; even its C-CDA export is mostly
  structured tables restating coded data, not dictated prose -- except the `DocumentReference`
  resources in this regenerated FHIR set, which do contain genuine synthetic clinical notes
  (Chief Complaint / HPI / Assessment and Plan). The C-CDA merge is still wired in and adds some
  value/diversity but is largely redundant with those; could be dropped later to cut pipeline
  runtime if it matters.
- No pipeline logic yet for `src/genomics_pipeline/`, `src/digital_twin/`, `src/fusion_layer/`,
  `src/governance/` (still empty packages).
- `chromadb` is installed but not yet wired into any code path.
- Clinical state vectors and wearable profiles are computed but not yet persisted to the database
  (`src/db/`) -- both pipelines only produce them in-memory (`run_emr_pipeline()`,
  `run_wearable_pipeline()`).
- The wearable pipeline's Docker image doesn't need any new dependencies (torch/numpy/scipy/
  neurokit2 were already in requirements.txt from Phase 0), but `tests/test_wearable_pipeline.py`
  hasn't been run inside a container yet -- only in the local venv.
- The "ouch meter" activation score is trained against a heuristic proxy target (see Phase 2
  completed notes above), not real pain/symptom self-reports -- there aren't any in this dataset.
  Treat the score as illustrative of the architecture, not a validated pain measure.
- Event-window segmentation falls back to session start/end when a session has no tags.csv
  entries (most "Final" sessions do); only some "Midterm 1"/"Midterm 2" sessions have real
  button-press-tagged events.

## Next Steps

- Phase 3: Genomics pipeline -- variant annotation, GWAS, GSEA pathway aggregation producing the
  genomic pathway profile
- Verify Phase 2 tests pass inside Docker (same as Phase 1's Docker verification)
- Persist EMR clinical state vectors and wearable profiles via `src/db/` (and/or chromadb) so later
  phases can read them back rather than recomputing
- Phase 4: Digital Twin Abstraction Layer combining the three embeddings per patient
- Phase 5: Generative Semantic Fusion Layer (Anthropic API, structured/schema-constrained output)
