# Progress

## Current Phase

Phase 8 -- Final Review

## Status

Complete -- all 8 phases complete

| Phase | Name | Status |
|---|---|---|
| 0 | Project scaffolding | Complete |
| 1 | EMR pipeline | Complete |
| 2 | Wearable pipeline | Complete |
| 3 | Genomics pipeline | Complete |
| 4 | Digital Twin Abstraction Layer | Complete |
| 5 | Generative Semantic Fusion Layer | Complete |
| 6 | Governance Layer | Complete |
| 7 | API + Demo UI | Complete |
| 8 | Final Review | Complete |

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

**Phase 3 -- Genomics pipeline** (architecture doc Section 3.3), for 5 VCF sample columns
(`HG00096`/`HG00097`/`HG00099`/`HG00100`/`HG00101`). **Correction (Phase 4):** these are *not*
the same patient identities as the 5 EMR patients or 5 wearable subjects -- see Phase 4's
completed notes below for why and how orchestration handles it.
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
- [x] `src/digital_twin/orchestrate.py` -- runs the three Layer 1 pipelines for one patient and
      calls `assemble_and_store_twin()` with their outputs. Surfaced a real data problem while
      building this: the EMR (Synthea), wearable (Exam Stress Dataset), and genomics (this
      demo's VCF) source datasets are three independent public datasets with **disjoint patient
      ID spaces** -- a Synthea `Patient.id` UUID, a wearable subject folder name ("S1".."S5"),
      and a VCF sample column ("HG00096" etc.) never refer to the same real person, so there is
      no natural shared `patient_id` across domains (see the corrected Phase 3 note above, which
      previously claimed otherwise). Asked the user how to handle this; the decision was to *not*
      pretend a unified ID exists: `orchestrate_twin_for_patient()` takes each domain's source ID
      explicitly (`emr_patient_id`, `wearable_subject_id`, `genomic_sample_id`) plus the
      `patient_id` to store the resulting twin under, so the correspondence is always an explicit
      choice made by the caller, never inferred by this module
- [x] `orchestrate_twins_for_cohort()` -- assembles twins for a list of explicit ID-mapping dicts,
      running the wearable and genomics pipelines exactly once and reusing their output for every
      mapping, rather than once per patient -- both are inherently cohort-level (one LSTM trained
      across all wearable subjects; GWAS/PCA/pathway enrichment computed across the whole VCF), so
      re-running them per patient would redundantly retrain/recompute the same cohort-wide result
- [x] `tests/test_digital_twin_orchestrate.py` -- end-to-end (no mocking, same convention as the
      other pipeline test files): runs the real EMR/wearable/genomics pipelines for one explicit
      ID triple and confirms the stored twin has all three embeddings at their expected
      dimensions, labeled with the correct pipeline_version, and round-trips correctly through
      `get_twin()`
- [x] Ran `orchestrate_twins_for_cohort()` for all 5 patients against the project's real
      `data/processed/twin.db`, with an explicit positional pairing across the three ID spaces
      declared at the call site (patient-1..5 <-> first..fifth EMR bundle <-> S1..S5 <->
      HG00096/97/99/100/101) -- an arbitrary demo convention, not a real identity claim. All 5
      twins stored at version 1 with correctly-dimensioned embeddings from all three domains
      (EMR 768, genomic 15, wearable 16) and correct per-domain pipeline_versions; verified via
      both `get_twin()` and `get_twin_domain()` for every patient and domain

**Phase 5 -- Generative Semantic Fusion Layer** (architecture doc Section 5, "Layer 3"), which
"performs cross-modal reasoning but never touches raw EMR text, genomic sequences, or physiological
signals directly -- it consumes only the structured embeddings produced by Layer 1"
- [x] `src/fusion_layer/formatting.py` -- `format_patient_summary()`/`format_cluster_summary()`
      turn one or more `DigitalTwin` rows into structured prompt text: per-domain summary
      statistics (dimensionality, L2 norm, mean, std) and traceability labels
      (`pipeline_version`, `embedding_id`) only, never the embeddings' own raw values (dumping
      hundreds of floats into a prompt isn't "structured text" the reasoning engine can use
      anyway) and never any raw domain data (which `DigitalTwin` rows never contain to begin
      with). `embedding_id()` (`"{patient_id}:v{version}:{domain}"`) is the canonical ID reused
      across chromadb entries, prompt text, and a hypothesis's `source_embedding_ids`, so all
      three can be cross-referenced
- [x] `src/fusion_layer/clustering.py` -- `cluster_twins()`: scikit-learn `KMeans` over each
      patient's concatenated (EMR+genomic+wearable) embedding -- the digital twin's own
      definition (architecture doc Section 4) -- standardized per-feature first so EMR's 768
      dimensions don't swamp genomic's ~15 and wearable's 16 in the distance metric.
      `n_clusters` capped at the sample count
- [x] `src/fusion_layer/vector_store.py` -- `upsert_twin_embeddings()`/`query_similar()`: each
      twin's three domain embeddings stored in chromadb via `PersistentClient` at
      `data/processed/chroma_db/` (gitignored, matches the doc's Section 9 "Digital Twin
      Layer | Storage | Versioned feature store / vector store" line), one collection per
      domain since EMR/genomic/wearable have different dimensionality and a chromadb
      collection expects uniform dims. Entries keyed by `embedding_id()`, so
      re-upserting a twin version is idempotent
- [x] `src/fusion_layer/reasoning.py` -- `generate_hypothesis()`: calls the real Claude API
      (`anthropic` SDK, `ANTHROPIC_API_KEY` read from `.env` via `get_anthropic_client()`,
      never hardcoded -- see Known Issues for a near-miss on this). Output forced into
      `HYPOTHESIS_JSON_SCHEMA` (`subgroup_trait`, `supporting_evidence`, `confidence`,
      `source_embedding_ids`) via Claude's tool-use API with `tool_choice` pinned to a single
      `record_hypothesis` tool, matching the doc's "Structured output enforcement" component.
      The prompt supplies a whitelist of candidate `source_embedding_ids` (the cluster's real
      chromadb entries) and `generate_hypothesis()` rejects (raises `ValueError`) any response
      citing an ID outside that whitelist -- a hallucinated ID would silently break the doc's
      "Traceability layer" guarantee ("Links every generated hypothesis back to the specific
      embedding evidence that produced it"), so it's enforced in code, not just prompted for.
      `_call_claude()` wraps the actual API call in `tenacity`-based retry with exponential
      backoff, retrying only transient failures (`APIConnectionError`, `APITimeoutError`,
      `RateLimitError`, `InternalServerError`) and never authentication/validation errors,
      which would just fail identically on every retry
- [x] `src/fusion_layer/models.py` -- `Hypothesis` SQLAlchemy model: `subgroup_trait`,
      `supporting_evidence` (JSON list), `confidence`, `source_embedding_ids` (JSON list) --
      `store_hypothesis()` re-validates against `HYPOTHESIS_JSON_SCHEMA` defensively before
      inserting, rather than trusting every caller already validated its input
- [x] `src/fusion_layer/pipeline.py` -- `run_fusion_layer_for_cohort()` orchestrates all of the
      above for a cohort: cluster twins, upsert every member's embeddings into chromadb,
      format each cluster, generate + store one hypothesis per cluster
- [x] `tests/test_fusion_layer.py` (12 tests) -- formatting labels every domain and never
      contains raw embedding values; clustering groups synthetically-separated patients
      correctly and caps `n_clusters`; chromadb upsert/query round-trips; `generate_hypothesis()`
      returns schema-valid JSON from a **mocked** Claude client (no live key needed to run the
      suite) and rejects both schema violations and hallucinated `source_embedding_ids`;
      `_call_claude()`'s retry logic verified directly (retries transient errors, does not retry
      `AuthenticationError`); every stored hypothesis has at least one traceable
      `source_embedding_id`, checked both right after storage and after a fresh query
- [x] Verified live against the real API (not just the mocked test suite): one direct
      `generate_hypothesis()` call, then a full `run_fusion_layer_for_cohort()` run against the
      5 real digital twins in `data/processed/twin.db` (Phase 4's output) -- 2 clusters,
      2 hypotheses stored, all 15 domain embeddings (5 patients x 3 domains) upserted into
      chromadb, every hypothesis's `source_embedding_ids` traced back to real chromadb entries
      (9 refs on one hypothesis, 6 on the other)
- [x] Found and fixed a real near-miss before starting this phase: `.env.example` (the
      template file meant to be committed to git) had been locally modified, uncommitted, to
      contain what looked like a real `ANTHROPIC_API_KEY` value, not a placeholder -- the
      committed version and the actual `.env` (gitignored) were both still empty/correct. Flagged
      it rather than continuing; the user moved the key to `.env` and cleared `.env.example`
      before this phase's work resumed. No leaked key ever reached git history

**Phase 6 -- Governance Layer** (architecture doc Section 7, cross-cutting): "Audit & traceability |
Every transformation and generative output is logged for post hoc review", "PHI separation", and
"Bias monitoring"
- [x] `src/governance/models.py` -- `AuditLogEntry` SQLAlchemy model: `timestamp`, `patient_id`
      (nullable -- a handful of actions are cohort-level, not about one patient), `pipeline_stage`,
      `action`, `source_file`, `pipeline_version`
- [x] `src/governance/audit.py` -- `log_audit_event()`: writes one row. If no `db` session is
      passed, opens its own short-lived session against the project's real database and, on that
      fallback path, lazily creates `audit_log` (`checkfirst=True`, idempotent) if it doesn't
      exist yet -- so every pipeline function below could call this with a single added line, no
      caller needed to separately provision the table or thread a session through first (this
      caught a real bug during development: without the lazy-create, adding this call broke the
      *existing*, unmodified `test_emr_pipeline.py`/`test_wearable_pipeline.py`/
      `test_genomics_pipeline.py`, none of which pass a `db`, since `audit_log` didn't exist yet
      in `data/processed/twin.db`)
- [x] Wired into every Phase 1-5 pipeline entry point -- one added `log_audit_event()` call at
      the end of each, no existing pipeline logic duplicated or changed:
      - `run_emr_pipeline_for_patient()` (`src/emr_pipeline/pipeline.py`) -- one row per patient
      - `run_wearable_pipeline()` (`src/wearable_pipeline/pipeline.py`) -- one row per subject
        (after the shared cohort-level LSTM training/scoring, since the doc's audit granularity
        is per-patient even though the underlying model run is cohort-wide)
      - `run_genomics_pipeline()` (`src/genomics_pipeline/pipeline.py`) -- one row per sample
        (same reasoning: cohort-level PLINK/SnpEff/gseapy run, per-sample audit rows)
      - `assemble_and_store_twin()` (`src/digital_twin/assembly.py`) -- one row per twin version,
        `pipeline_version` recorded as all three source pipeline versions joined
        (`"emr=...;genomic=...;wearable=..."`, since one twin draws on three)
      - `store_hypothesis()` (`src/fusion_layer/reasoning.py`) -- one row per patient actually
        cited in the hypothesis's `source_embedding_ids` (parsed from the
        `"{patient_id}:v{version}:{domain}"` IDs), `pipeline_version` set to the Claude model used
      - Each of `run_emr_pipeline_for_patient`/`run_wearable_pipeline`/`run_genomics_pipeline`
        gained an optional `db: Session | None = None` parameter (default preserves every
        existing call site's behavior unchanged); `assemble_and_store_twin`/`store_hypothesis`
        already took `db` as a required parameter from Phase 4/5, so no signature change there
      - Avoided a real circular-import hazard while wiring this up: `src/governance/phi_check.py`
        needs to import from `src.emr_pipeline` (reusing its PHI definitions), while
        `src.emr_pipeline`'s pipeline functions need to import `log_audit_event` from
        `src.governance` -- eagerly importing `phi_check` at `src/governance/__init__.py`'s top
        level would make the two packages import each other. Fixed by not importing `phi_check`
        at the package root (only `audit.py`/`fairness_check.py`/`models.py`, none of which
        depend on any other phase); callers needing PHI checking import
        `from src.governance.phi_check import ...` directly. Verified with several import
        orderings, not just the one that happened to work first
- [x] `src/emr_pipeline/deidentify.py` -- renamed the module-private `_get_engines()` to a
      public `get_presidio_engines()` so `phi_check.py` can reuse the same cached
      (`@lru_cache`) Presidio engine instance rather than constructing (and loading
      `en_core_web_lg` into) a second one
- [x] `src/governance/phi_check.py` -- `check_demographics_for_phi()`/`check_notes_for_phi()`/
      `check_patient_record_for_phi()`: reuses Phase 1's own PHI definitions
      (`PHI_DEMOGRAPHIC_FIELDS`, `PHI_EXTENSION_URLS`, `PHI_ENTITIES`) and cached Presidio engine
      rather than redefining what counts as PHI a second time. Demographics check: any
      `PHI_DEMOGRAPHIC_FIELDS` key or `PHI_EXTENSION_URLS` extension still present should have
      been dropped outright by Phase 1's `scrub_demographics()`. Notes check: a fresh Presidio
      re-scan of already-processed note text for the same `PHI_ENTITIES` -- independent of, not a
      re-run of, Phase 1's own deny-list-plus-Presidio de-identification
- [x] `src/governance/fairness_check.py` -- `check_subgroup_fairness()`: a fairlearn `MetricFrame`
      computing a metric per subgroup plus the max difference/ratio across subgroups -- the
      standard subgroup-disparity summary fairlearn provides out of the box. Explicitly a stub, not
      a validated bias finding: this project's 5-patient cohort is definitionally too small for
      any subgroup statistic to mean anything (one patient's value alone can swing a whole
      subgroup's mean); `n_per_group`/`small_sample_warning` (vs. a documented
      `MIN_MEANINGFUL_GROUP_SIZE` floor) surface that limitation explicitly rather than silently
      reporting numbers a reader could mistake for meaningful. No ground-truth outcome labels
      exist in this demo either, so `y_true` defaults to `y_pred` itself and the demonstration
      metric is a plain per-group mean -- real usage means wiring in real outcome
      labels/predictions and real demographic strata (age, sex, ancestry, severity, per the doc)
      once cohort size is large enough
- [x] `tests/test_governance.py` (9 tests) -- one audit-log-entry test per Phase 1-5 pipeline
      entry point (EMR/wearable/genomics run for real, digital twin assembly and fusion layer
      hypothesis generation use lightweight/mocked inputs matching each phase's own test
      convention); PHI check flags a deliberately-inserted PHI value and passes on clean
      synthetic data; a third, stronger PHI check test runs the real EMR pipeline end-to-end and
      confirms no *actual* patient identifier leaked into any check finding (see below for why it
      doesn't assert zero findings); fairness check stub runs and correctly flags the small-sample
      warning
- [x] A real, interesting finding surfaced while writing the third PHI check test: Presidio's
      fresh re-scan of already-de-identified note text flagged a literal `http://snomed.info/sct`
      coding-system URL as `PERSON`, repeatedly -- not real PHI. Root cause: Phase 1's own
      anonymization already replaced nearby content with placeholder tags like `<US_SSN>`/
      `<DATE_TIME>`, and those placeholder tags change the surrounding text enough to trigger a
      *new* false-positive NER match that wasn't present in the original pass (confirmed by
      inspecting the actual note text: `'...Code: http://snomed.info/sct <US_SSN>. Start:...'`).
      This is the independent post-hoc check correctly re-verifying downstream output -- exactly
      what it's for -- not a bug in `phi_check.py` or in Phase 1's de-identification. Left the
      check's logic as-is (a faithful reuse of Presidio/Phase 1's PHI definitions) rather than
      adding suppression logic to hide it; the test instead asserts that no finding's flagged text
      matches this patient's actual known identifiers, which passes. See Known Issues.

**Phase 7 -- API + Demo UI**: read-only FastAPI access to the digital twin store (Phase 4) and the
fusion layer's generated hypotheses (Phase 5), plus a Streamlit UI over that API -- "the doctor-
facing view" the task described. Nothing in either layer runs a pipeline or calls the Claude API on
request; both only ever serve/display already-computed, already-stored data
- [x] `api/schemas.py` -- Pydantic response models kept separate from routing logic:
      `FullTwinResponse` (nests each domain's `pipeline_version`+`embedding` under its own key,
      so a domain-filtered response is structurally a strict subset of the full-twin response,
      not just conventionally one), `DomainResponse` (`patient_id`, `version`, `domain`,
      `pipeline_version`, `embedding` -- no field that could ever carry another domain's data),
      `HypothesisResponse`/`HypothesesResponse`, `PatientsResponse`
- [x] `api/main.py` -- `GET /patients` (every patient_id with a stored twin -- what the UI's
      picker populates itself from), `GET /patient/{id}/full-twin`, `GET /patient/{id}/{emr,
      genomic,wearable}` (404 if no twin exists for that patient), `GET /patient/{id}/hypotheses`
      (200 + empty list for a patient with none -- that's a valid result, not an error). A
      `lifespan` handler now calls `Base.metadata.create_all(bind=engine)` on startup -- the API
      is the first part of the running application (as opposed to a test fixture or one-off
      verification script) that needs the digital_twins/hypotheses/audit_log tables to already
      exist, so this finally closes a gap every earlier phase's tests/scripts had to work around
      individually
- [x] `src/digital_twin/retrieval.py` -- added `list_patient_ids()` (distinct patient_ids with a
      stored twin); `src/fusion_layer/retrieval.py` -- added `get_hypotheses_for_patient()`,
      matching by parsing each `source_embedding_ids` entry's leading `"{patient_id}:..."`
      component (there's no `patient_id` column on `Hypothesis` since one hypothesis can span
      several patients) rather than a database-level JSON query, portable across SQLite/Postgres
- [x] `app/main.py` -- Streamlit UI: a patient picker (populated from `/patients`, not
      hardcoded); a "Digital Twin" tab with nested tabs ("All domains"/"EMR only"/"Genomic
      only"/"Wearable only") rendering each domain's embedding as a line chart labeled with its
      pipeline_version; a "Hypotheses" tab listing every hypothesis for the selected patient with
      its supporting evidence and a `source_embedding_ids` traceability block. Talks to the API
      over HTTP (`requests`, `API_BASE_URL` env var, defaults to `http://localhost:8000`) rather
      than importing `src/` or querying the database directly, so the UI only ever sees what the
      API actually serves -- the same contract any other API client would have
- [x] `docker-compose.yml` -- added a `streamlit` service (same image as the existing `app`
      service, built from the same `docker/Dockerfile`, just a different `command:` override --
      no second Dockerfile needed) on port 8501, reaching the API over the compose network at
      `http://app:8000` (Docker's internal DNS resolves the service name). Kept the existing `app`
      service's name as-is rather than renaming it to `api` -- avoided unnecessary churn against
      already-documented commands (`docker compose run --rm app pytest`, used throughout
      PROGRESS.md and README.md) for a purely cosmetic rename
- [x] `tests/test_api.py` (13 tests) -- against an isolated in-memory SQLite database
      (`StaticPool` + `check_same_thread=False`, needed because `TestClient` dispatches route
      handlers on a worker thread, not the test's own thread -- a bare in-memory SQLite
      connection is both thread-bound and, per connection, its own separate empty database)
      wired in via FastAPI's `dependency_overrides`: every endpoint returns 200 with
      correctly-shaped data against a twin whose three domains carry distinct,
      individually-recognizable embeddings; each domain endpoint's response never contains
      another domain's key, embedding, or pipeline_version (checked positively -- the returned
      values are compared against the *other* domains' actual seeded values, not just "some
      value present"); 404s for an unknown patient; hypotheses endpoint returns the seeded
      hypothesis with its `source_embedding_ids` intact, and 200 + empty list (not 404) for a
      patient with none yet
- [x] Verified end-to-end for real, not just via pytest: built and ran both Docker services
      (`docker compose up -d`), confirmed `/health` and `/patients` respond correctly from the
      host and the Streamlit UI at `localhost:8501` is reachable, then drove the actual UI in a
      browser -- selected different patients, confirmed each domain-only tab shows only that
      domain's chart, confirmed the Hypotheses tab shows patient-1/2/4's real Phase 5 hypothesis
      (9 source_embedding_ids) when patient-1 is selected and patient-3/5's different hypothesis
      (6 source_embedding_ids, correctly *not* patient-1's) when patient-3 is selected. No
      console errors. Stopped both containers afterward (`docker compose down`)

**Phase 8 -- Final Review**
- [x] Ran the full pytest suite (all nine test files, covering all seven working phases) in both
      local venv and Docker. All 87 tests passed in both -- no failures found, so there was
      nothing to fix
- [x] Rewrote `README.md` end to end as the project's front door: what this project is, an
      architecture overview table, a `docker compose up --build` quick start with the exact URLs
      for the API and the Streamlit UI, consolidated data-placement instructions (what's
      committed vs. what needs to be downloaded/generated and where it goes), local (non-Docker)
      setup, the API reference, and the test-running instructions -- reorganizing and tightening
      material that had accreted phase-by-phase rather than dropping any of the operational
      detail those phases established
- [x] Added a "Migrating to PostgreSQL later" section to the README: uncomment the `postgres`
      service block in `docker-compose.yml`, set `DATABASE_URL` in `.env` to a PostgreSQL
      connection string, `docker compose up --build` -- no code changes, since `src/db/session.py`
      already builds its engine from whatever `DATABASE_URL` it finds and every model uses
      portable SQLAlchemy column types
- [x] Added a phase-completion overview table at the top of this file (all 8 phases, all
      Complete) -- the detailed per-phase logs below are unchanged, this is a quick-glance summary
      on top of them, not a replacement

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

Local venv, full suite including the new orchestration test (2026-09-01):
```
53 passed, 1 warning in 143.21s (0:02:23)
```
`tests/test_digital_twin_orchestrate.py` (1 test) -- PASSED. Runs the real EMR/wearable/genomics
pipelines end-to-end for one explicit patient/subject/sample ID triple (no mocking), against an
isolated in-memory SQLite DB.

Docker (`docker compose build` then `docker compose run --rm app pytest -q`, 2026-09-01). No new
system dependencies needed for this phase:
```
53 passed in 374.98s (0:06:14)
```
All 53 tests, across all six test files, pass in both environments. The Docker run is noticeably
slower than local venv (375s vs. 143s) here specifically because `test_digital_twin_orchestrate.py`
now runs the full wearable cohort pipeline (150-epoch LSTM training, CPU-only in the container) in
addition to the already-CPU-bound EMR Bio_ClinicalBERT inference -- both now happen in the same
test run rather than across separate test files.

Also ran `orchestrate_twins_for_cohort()` outside pytest, for all 5 patients against the project's
real `data/processed/twin.db` -- see Phase 4 completed notes above for the result.

Local venv, full suite including the new fusion layer tests (2026-09-01):
```
65 passed, 1 warning in 148.40s (0:02:28)
```
`tests/test_fusion_layer.py` (12 tests) -- PASSED. No live `ANTHROPIC_API_KEY` needed to run the
suite -- `generate_hypothesis()` is exercised against a mocked `anthropic.Anthropic` client
throughout.

Docker (`docker compose build` then `docker compose run --rm app pytest -q`, 2026-09-01). No new
system dependencies (`scikit-learn`/`tenacity`/`jsonschema` are pure-Python, picked up by the
existing `pip install -r requirements.txt` step):
```
65 passed in 308.46s (0:05:08)
```
All 65 tests, across all seven test files, pass in both environments.

Also verified live against the real Claude API (outside pytest, `ANTHROPIC_API_KEY` from `.env`):
one direct `generate_hypothesis()` call, then `run_fusion_layer_for_cohort()` for all 5 real
digital twins -- see Phase 5 completed notes above.

Local venv, full suite including the new governance tests (2026-09-02):
```
74 passed, 1 warning in 203.69s (0:03:23)
```
`tests/test_governance.py` (9 tests) -- PASSED.

Docker (`docker compose build` then `docker compose run --rm app pytest -q`, 2026-09-02). No new
system dependencies (`fairlearn` is pure-Python, picked up by the existing
`pip install -r requirements.txt` step):
```
74 passed in 543.63s (0:09:03)
```
All 74 tests, across all eight test files, pass in both environments. Slower than local venv here
specifically because `test_governance.py`'s audit-log tests re-run the real EMR/wearable/genomics
pipelines (CPU-bound in the container, same as `test_emr_pipeline.py` etc.) on top of everything
those files already run.

Note: because `run_emr_pipeline_for_patient`/`run_wearable_pipeline`/`run_genomics_pipeline`'s
audit logging falls back to the real `data/processed/twin.db` whenever no `db` is passed (see
Phase 6 completed notes above), and `tests/test_{emr,wearable,genomics}_pipeline.py` were
deliberately left unmodified (per this phase's "don't touch existing pipeline logic" instruction)
rather than updated to inject an isolated session, every full-suite test run appends real rows to
that database's `audit_log` table -- by design (an audit log is supposed to capture every real
invocation, tests included), not a leak or a bug, but worth knowing if `audit_log`'s row count
looks larger than "5 patients x however many phases" would suggest.

Local venv, full suite including the new API tests (2026-09-02):
```
87 passed, 1 warning in 203.23s (0:03:23)
```
`tests/test_api.py` (13 tests) -- PASSED.

Docker (`docker compose build` then `docker compose run --rm app pytest -q`, 2026-09-02). No new
system dependencies (`requests` is pure-Python, picked up by the existing
`pip install -r requirements.txt` step):
```
87 passed in 535.44s (0:08:55)
```
All 87 tests, across all nine test files, pass in both environments.

Also verified for real, outside pytest: both Docker services running together (`docker compose up
-d`), API endpoints checked via `curl` from the host, and the Streamlit UI driven directly in a
browser -- patient picker, all three domain-filter tabs, and both real Phase 5 hypotheses (correctly
different per patient) all confirmed rendering correctly with no console errors. See Phase 7
completed notes above for detail. Stopped both containers afterward.

**Final full-suite run (Phase 8, 2026-09-02)** -- all 87 tests, across all nine test files, no
changes since Phase 7's run:

Local venv:
```
87 passed, 1 warning in 211.13s (0:03:31)
```

Docker (`docker compose build` then `docker compose run --rm app pytest -q`):
```
87 passed in 443.89s (0:07:23)
```

No failing tests found -- nothing needed fixing.

## Known Issues / Blockers

- The EMR, wearable, and genomics source datasets have **disjoint patient ID spaces** (Synthea
  `Patient.id` UUIDs / wearable "S1".."S5" / VCF sample columns "HG00096" etc.) -- there is no real
  correspondence between "patient 1" in one domain and "patient 1" in another. Every twin assembled
  so far (in tests and in the real `data/processed/twin.db`) uses an arbitrary positional pairing
  chosen at the call site (see Phase 4 completed notes above), not a genuine identity link. Treat
  `data/processed/twin.db`'s current 5 rows as a mechanics demonstration, not 5 real patients'
  actual combined state.
- `orchestrate_twin_for_patient()` re-runs the *entire* wearable and genomics cohort pipelines
  (LSTM training across all subjects; PLINK/SnpEff/gseapy across the whole VCF) on every call
  unless `wearable_profiles`/`genomic_profiles` are passed in from a previous run -- calling it in
  a loop for multiple patients without doing that is wasteful. `orchestrate_twins_for_cohort()`
  avoids this by construction; prefer it whenever assembling more than one patient's twin.
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
- Presidio's PHI re-scan (`src/governance/phi_check.py`) can flag benign false positives on
  already-processed note text that its own (or Phase 1's) anonymization placeholder tags
  introduced -- e.g. a literal `http://snomed.info/sct` coding-system URL getting misclassified
  as `PERSON` once a neighboring `<US_SSN>`/`<DATE_TIME>` tag changes local context (see Phase 6
  completed notes above for the confirmed root cause). This is the check correctly catching
  *something*, just not real PHI -- always check a finding's flagged snippet against the
  patient's actual known identifiers (`known_identifier_strings()`) before treating a `passed:
  False` result as a genuine leak, exactly as `tests/test_governance.py`'s third PHI check test
  does.
- No cross-cutting audit-logging/institutional-boundary enforcement beyond what Phase 6 built
  (audit logging + PHI/fairness check stubs) -- "institutional boundaries" (doc: "local data
  stays within institutional boundaries; only embeddings/model updates move", e.g. federated
  learning frameworks) is still entirely unaddressed, since this demo has always run as a single
  local instance with no multi-institution data-sharing scenario to enforce boundaries between.
- The 2 hypotheses currently stored in `data/processed/twin.db`'s `hypotheses` table were
  generated from that same database's 5 digital twins, which (see the ID-space issue
  above) are an arbitrary positional pairing across three unrelated datasets, not real linked
  patients -- so these specific hypotheses are a demonstration of the mechanism, not a claim
  about any real cross-modal pattern. Regenerate once/if real linked patient data exists.
- The `.env.example` near-miss (see Phase 5 completed notes above) is resolved for this repo,
  but worth restating: never edit `.env.example` with a real value, even temporarily -- put
  real secrets only in `.env` (gitignored). If the key that appeared there was ever real and
  used elsewhere, consider rotating it in the Anthropic console regardless.
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
- A real cross-domain patient identity mapping, if/when one becomes available -- the current
  positional pairing (see Known Issues) is a demo convenience, not something to build Phase 5
  reasoning claims on top of as if it were real linked patient data
- Institutional-boundary enforcement (doc Section 7: "local data stays within institutional
  boundaries; only embeddings/model updates move", e.g. federated learning frameworks) -- not
  addressed by Phase 6's audit logging/PHI check/fairness stub, and genuinely out of scope for a
  single-instance local demo; would only become relevant if this ever ran across more than one
  institution's data
- A real bias-monitoring run once cohort size is large enough for `check_subgroup_fairness()`'s
  numbers to mean something (see Phase 6 completed notes and Known Issues -- `MIN_MEANINGFUL_GROUP_SIZE`
  is never met at 5 patients), with real outcome labels/predictions and real demographic strata
  in place of the current stub's placeholder `y_true=y_pred` and synthetic subgroup splits
- Phase 7's API is read-only -- viewing hypotheses is wired into the app now, but *triggering* a
  pipeline run or a new fusion-layer hypothesis generation still requires direct Python calls
  (`orchestrate_twins_for_cohort()`, `run_fusion_layer_for_cohort()`), not anything reachable
  through `api`/`app`. A `POST` endpoint (or an admin action in the Streamlit UI) to kick off
  either would close that gap, but needs its own thinking about request duration (the wearable/
  genomics cohort pipelines take minutes) and about not exposing `ANTHROPIC_API_KEY` usage to
  arbitrary API callers without some access control first (see below).
- `AuditLogEntry` still has no read endpoint (`Hypothesis` now does, via `/patient/{id}/hypotheses`)
  -- the "post hoc review" the doc's audit-logging concern is actually for still means direct
  SQLAlchemy access to `data/processed/twin.db`'s `audit_log` table, not anything through the
  running application.
- No authentication/access control on any API endpoint -- fine for a local single-user demo, but
  worth flagging explicitly before this is ever reachable from anywhere other than localhost/a
  trusted Docker network, since it currently serves patient-level data (synthetic, but still
  structured the way real patient data would be) to any caller that can reach port 8000.
