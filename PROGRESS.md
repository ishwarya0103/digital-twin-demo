# Progress

## Current Phase

Phase 4 -- Digital Twin Abstraction Layer

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

**Phase 3 -- Genomics pipeline** (architecture doc Section 3.3), for the same 5 sample IDs used
across the EMR/wearable phases' patient identifier convention
- [x] `src/genomics_pipeline/vcf_loader.py` -- reads sample IDs and a per-variant, per-sample
      genotype dosage table from a single VCF discovered by extension (`*.vcf`/`*.vcf.gz`) under
      `data/raw/genomics/`; resolves the discovered path to absolute so it survives the
      working-directory changes of the PLINK/SnpEff subprocess calls downstream
- [x] `src/genomics_pipeline/config.py` -- resolves the PLINK and SnpEff binaries (PATH, then
      `PLINK_BIN`/`SNPEFF_BIN` env var overrides, then a local conda env fallback); SnpEff stands in
      for ANNOVAR (gated behind manual registration, no package-manager install path)
- [x] `src/genomics_pipeline/annotation.py` -- Stage 1 (variant annotation). Shells out to SnpEff
      (GRCh37.75) to label each variant with gene/functional consequence
- [x] `src/genomics_pipeline/gwas.py` -- Stage 2 (GWAS) and Stage 3 (population stratification), both
      via PLINK: `--assoc` against a synthetic alternating case/control phenotype (this cohort has no
      real clinical trait to test against) and `--pca` for ancestry components. With 5 samples the
      association test is statistically underpowered by construction -- illustrative of pipeline
      mechanics, not evidence of real association
- [x] `src/genomics_pipeline/pathway_aggregation.py` -- Stage 4 (pathway-level aggregation) via
      gseapy's `gp.enrich()` (hypergeometric over-representation, not permutation-based ranked GSEA --
      too few annotated genes here for that to be stable) against `pain_pathways.gmt`, a small
      KEGG-derived gene-set collection (inflammation, nociception, drug metabolism) fetched once from
      Enrichr's KEGG_2021_Human library. Handles gseapy returning `[]` instead of an empty DataFrame
      when none of the hit genes overlap any pathway gene set at all (as opposed to overlapping some
      pathways but not others) -- the case this project's tiny synthetic VCF actually hits
- [x] `src/genomics_pipeline/embedding.py` -- Stage 5. Builds each patient's genomic pathway profile
      vector (pathway enrichment scores + ancestry PCs)
- [x] `src/genomics_pipeline/pipeline.py` -- orchestrates all five stages; resolves `work_dir` to an
      absolute path up front so the PLINK output-file chain (bed/pheno/assoc/eigenvec paths, all
      derived from `work_dir`) stays consistent across the subprocess calls that change `cwd`
- [x] `tests/test_genomics_pipeline.py` -- PLINK runs and PCA covers all 5 samples, gseapy runs
      without error, all 5 samples processed, pathway-score vector shape and traceability fields
      correct per sample
- [x] `data/raw/genomics/5patients_test.vcf.gz` -- a small hand-crafted, valid VCF (10 variants on
      chr21, real VCF 4.2 format, BGZF-compressed, synthetic genotypes) for the same 5 sample IDs
      used elsewhere in this phase. **Not real 1000 Genomes data.** The public 1000 Genomes FTP
      mirrors were unreachable and a guessed GitHub test-file fallback also failed, so this
      hand-crafted substitute exercises the pipeline mechanics (VCF parsing, PLINK GWAS/PCA, SnpEff
      annotation, gseapy enrichment) end-to-end without making any claim about real genetic
      association or ancestry for these sample IDs

**Phase 4 -- Digital Twin Abstraction Layer** (architecture doc Section 4, "Layer 2"), combining
each patient's three Layer 1 embeddings into one versioned, traceable record -- concatenation and
storage only, no reasoning/prediction/transformation, per the doc's explicit scope boundary for
this layer
- [x] `src/digital_twin/models.py` -- `DigitalTwin` SQLAlchemy model, one row per patient per
      version (`UniqueConstraint("patient_id", "version")`). Each domain gets its own embedding
      column *and* its own `..._pipeline_version` column (`emr_embedding`/`emr_pipeline_version`,
      `genomic_embedding`/`genomic_pipeline_version`, `wearable_embedding`/
      `wearable_pipeline_version`) so every twin component stays traceable to its source domain
      and the pipeline version that produced it, per the doc's Layer 2 design-choice table.
      Embeddings stored as JSON (`list[float]`) rather than raw binary -- portable across SQLite
      and the Postgres migration path with no code changes, matching `src/db/`'s existing
      `DATABASE_URL`-only-swap convention
- [x] `src/digital_twin/assembly.py` -- `assemble_and_store_twin()` takes a patient's three
      already-computed embeddings (plus each one's pipeline_version) and writes a new row with
      `version` = that patient's current highest version + 1 (1 for a first twin). Never updates
      or deletes a prior row -- every previously assembled twin for a patient stays queryable
      exactly as it was. Deliberately takes the three embeddings as direct arguments rather than
      invoking the EMR/wearable/genomics pipelines itself, keeping this layer decoupled from how
      each Layer 1 pipeline is actually invoked (batch cohort run vs. per-patient), matching the
      doc's "batch/retrospective" update-frequency design choice
- [x] `src/digital_twin/retrieval.py` -- `get_twin()` (full record, latest version by default or
      a specific `version`) and `get_twin_domain()` (one domain's embedding + pipeline_version
      only, `domain` in `{"emr", "genomic", "wearable"}`, raises on an unrecognized domain)
- [x] `tests/test_digital_twin.py` -- against an isolated in-memory SQLite DB (not the project's
      real `data/processed/twin.db`): a stored twin contains all three embeddings correctly
      labeled by source and pipeline_version; assembling a second version for the same patient
      leaves the first version's row unchanged and queryable (both rows coexist, latest-version
      lookup returns the newer one); domain-filtered retrieval returns only the requested
      domain's embedding/pipeline_version, no keys from the other two domains; unknown-domain and
      unknown-patient edge cases

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
subjects) -- PASSED.

Docker (`docker compose build` then `docker compose run --rm app pytest -v`, 2026-08-26). No new
image layers needed -- torch/numpy/scipy/neurokit2 were already baked in from Phase 0, so the
rebuild only picked up the new source files via `COPY . .`; `data/raw/wearable/` (gitignored, not
in the image) is supplied at runtime by the same `./data:/app/data` volume mount that already
carries `data/raw/emr/`:
```
33 passed, 1 warning in 175.14s (0:02:55)
```
All 33 tests, across all three test files, pass in both environments.

Local venv, full suite including the new genomics tests (2026-09-01):
```
47 passed, 1 warning in 106.91s (0:01:46)
```
`tests/test_genomics_pipeline.py` (14 tests) -- PASSED. PLINK and SnpEff resolved via the local
conda env fallback in `config.py` (`~/miniforge/.../envs/genomics/bin`), not PATH.

Docker (`docker compose build` then `docker compose run --rm app pytest -v`, 2026-09-01). Added
`plink1.9`, `snpeff`, and `default-jre-headless` to `docker/Dockerfile` via `apt-get` (the Debian
`snpeff` package installs the binary as `/usr/bin/snpEff`; `plink1.9` installs as
`/usr/bin/plink1.9`, not `plink` -- `PLINK_BIN`/`SNPEFF_BIN` env vars set in the Dockerfile point
`config.py`'s binary resolution at both explicitly). The SnpEff GRCh37.75 genome database is not
fetched during the Docker build (its usual source, `snpeff.blob.core.windows.net`, was unreachable
from this project's build environment -- same public-mirror situation as the VCF itself); instead
`data/raw/genomics_snpeff_db/GRCh37.75/` (trimmed to chr21-only, ~106MB, gitignored -- see Genomics
data in README.md) is supplied at runtime through the existing `./data:/app/data` volume mount, the
same way `data/raw/wearable/` already works, and `annotation.py` passes it to SnpEff explicitly via
`-dataDir` (added `SNPEFF_DATA_DIR` in `config.py`) rather than relying on SnpEff's own default data
directory, which differs between the local conda install and the Docker image:
```
47 passed in 274.18s (0:04:34)
```
`tests/test_genomics_pipeline.py` (14 tests) -- PASSED in Docker, matching local venv results.
All 47 tests, across all four test files, now pass in both environments.

Local venv, full suite including the new digital twin tests (2026-09-01):
```
52 passed, 1 warning in 104.05s (0:01:44)
```
`tests/test_digital_twin.py` (5 tests) -- PASSED. Runs against an isolated in-memory SQLite
database created per test, not the project's real `data/processed/twin.db`.

Docker (`docker compose build` then `docker compose run --rm app pytest -q`, 2026-09-01). No new
system dependencies needed for this phase (`src/digital_twin/` only uses SQLAlchemy, already in
the image) -- rebuild only picked up the new source files via `COPY . .`:
```
52 passed in 240.41s (0:04:00)
```
All 52 tests, across all five test files, pass in both environments.

## Known Issues / Blockers

- `assemble_and_store_twin()` takes each domain's embedding directly rather than pulling it from
  `run_emr_pipeline`/`run_wearable_pipeline`/`run_genomics_pipeline` itself -- there's no
  orchestration function yet that runs all three Layer 1 pipelines for one patient and calls
  `assemble_and_store_twin()` with their outputs. `api`/`app` (or a future phase) will need that
  glue when the digital twin actually gets populated from real pipeline runs rather than tests.

- `data/raw/genomics/5patients_test.vcf.gz` is a hand-crafted synthetic VCF, not real 1000 Genomes
  data -- the public FTP mirrors were unreachable when this phase was built (see Phase 3 completed
  notes above). Its 10 variants don't fall in genes covered by `pain_pathways.gmt`'s curated gene
  sets, so pathway enrichment scores are neutral (p=1.0) for every sample; this is expected given the
  placeholder data, not a pipeline bug -- swap in real 1000 Genomes data later to get non-trivial
  enrichment results.
- `data/raw/genomics_snpeff_db/GRCh37.75/` (SnpEff's annotation database) is trimmed to chr21 only,
  matching the demo VCF -- annotating a VCF with variants on any other chromosome will fail against
  this local copy until the corresponding `sequence.<chrom>.bin` is added. See README.md's Genomics
  data section.

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
- No pipeline logic yet for `src/fusion_layer/`, `src/governance/` (still empty packages).
- `chromadb` is installed but not yet wired into any code path.
- The three Layer 1 pipelines (`run_emr_pipeline()`, `run_wearable_pipeline()`,
  `run_genomics_pipeline()`) still only produce their embeddings in-memory -- `src/digital_twin/`
  can now store and version them once assembled, but nothing yet calls all three pipelines for one
  patient and feeds their outputs into `assemble_and_store_twin()` (see Phase 4 completed notes
  above and Next Steps).
- The "ouch meter" activation score is trained against a heuristic proxy target (see Phase 2
  completed notes above), not real pain/symptom self-reports -- there aren't any in this dataset.
  Treat the score as illustrative of the architecture, not a validated pain measure.
- Event-window segmentation falls back to session start/end when a session has no tags.csv
  entries (most "Final" sessions do); only some "Midterm 1"/"Midterm 2" sessions have real
  button-press-tagged events.

## Next Steps

- Swap the hand-crafted `data/raw/genomics/5patients_test.vcf.gz` for a real 1000 Genomes subset
  once a working source is found, so pathway enrichment produces non-trivial scores -- if the
  replacement covers more than chr21, `data/raw/genomics_snpeff_db/GRCh37.75/` needs the matching
  additional `sequence.<chrom>.bin` file(s) too
- Orchestration glue that runs all three Layer 1 pipelines for a given patient_id and calls
  `assemble_and_store_twin()` with their outputs -- `src/digital_twin/` can store and version
  twins, but nothing yet wires real pipeline runs into it end-to-end (see Known Issues)
- Phase 5: Generative Semantic Fusion Layer (Anthropic API, structured/schema-constrained output)
