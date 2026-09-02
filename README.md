# Digital Twin Demo

A small (5-patient) working demo of a "light digital twin" architecture for precision medicine,
described in *A Scalable Generative AI Architecture for Lightweight Digital Twins in Precision
Medicine and Drug Discovery* and its accompanying `Full_Architecture_and_Tech_Stack.pdf` reference.

Three independent pipelines turn raw EMR, genomic, and wearable data into structured, per-patient
embeddings. Those three embeddings are combined into one versioned "digital twin" record per
patient, and a generative semantic fusion layer (Claude) reasons across clusters of twins to
produce structured, traceable research hypotheses -- each one linked back to the exact embeddings
that produced it. A cross-cutting governance layer logs every pipeline run, checks processed EMR
data for leftover PHI, and includes a bias-monitoring stub. A FastAPI backend and a Streamlit UI
expose all of this for browsing.

The system is **retrospective and advisory only** -- it does not perform real-time clinical
decision-making, and no output here is a diagnosis, prediction, or treatment recommendation.

See [PROGRESS.md](PROGRESS.md) for the full phase-by-phase build log, test results, and known
limitations.

## Architecture

```
EMR pipeline -------\
Wearable pipeline ----> Digital Twin (per patient, versioned) --> Fusion Layer --> Hypotheses
Genomics pipeline --/                                             (Claude, clustered)

Governance layer (audit log, PHI check, fairness stub) cuts across all of the above.
API (FastAPI) reads the twin store + hypotheses  -->  Streamlit UI
```

| Layer | Code | What it does |
|---|---|---|
| EMR pipeline | `src/emr_pipeline/` | De-identifies + extracts a clinical state vector from Synthea FHIR data |
| Wearable pipeline | `src/wearable_pipeline/` | Wearable sensor time-series -> physiological state profile ("ouch meter") |
| Genomics pipeline | `src/genomics_pipeline/` | VCF -> GWAS + pathway aggregation -> genomic pathway profile |
| Digital Twin | `src/digital_twin/` | Combines the three embeddings into one versioned per-patient record |
| Fusion Layer | `src/fusion_layer/` | Clusters twins, calls Claude for structured, traceable hypotheses |
| Governance | `src/governance/` | Audit logging, PHI re-check, fairness-check stub |
| API | `api/` | FastAPI, read-only access to twins and hypotheses |
| UI | `app/` | Streamlit, browses the API |

## Quick start (Docker)

```bash
cp .env.example .env          # fill in ANTHROPIC_API_KEY if you want to regenerate hypotheses
docker compose up --build
```

This builds one image and starts two services from it:

- **API** (FastAPI) at [http://localhost:8000](http://localhost:8000) -- see `/docs` for the
  interactive OpenAPI UI
- **Demo UI** (Streamlit) at [http://localhost:8501](http://localhost:8501) -- open this in a
  browser once both containers are up; it talks to the API over the Docker network, not the
  database directly

`./data` is mounted into both containers, so the SQLite database at `data/processed/twin.db`
persists across restarts and is shared between the two services. If you already have raw pipeline
data and a populated `data/processed/twin.db` from a previous run (or from following the Data
section below), the UI will show real patients and hypotheses immediately; otherwise it'll come up
empty until you run the pipelines (see [PROGRESS.md](PROGRESS.md) for how each phase does that).

To stop everything: `docker compose down`.

## Project layout

```
src/emr_pipeline/       EMR extraction -> clinical state vector
src/wearable_pipeline/  Wearable time-series -> physiological state profile ("ouch meter")
src/genomics_pipeline/  GWAS + pathway aggregation -> genomic pathway profile
src/digital_twin/       Digital Twin Abstraction Layer (combines the three embeddings)
src/fusion_layer/       Generative Semantic Fusion Layer
src/governance/         De-identification checks, audit logging, fairness stub
src/db/                 SQLAlchemy engine/session, swappable via DATABASE_URL
api/                    FastAPI app
app/                    Streamlit UI
data/raw/emr/            Synthea FHIR R4 Bundle JSON, one file per patient (filenames as
                         downloaded from Synthea's output/fhir/, unchanged)
data/raw/emr_notes/      Matching Synthea C-CDA XML per patient (output/ccda/, same generation
                         run as emr/ so patient UUIDs line up), optional
data/raw/emr_metadata/   Synthea's hospitalInformation*/practitionerInformation*.json --
                         not patient records, kept out of the pipeline's input directory
data/raw/wearable/       Wearable Exam Stress Dataset (Empatica E4 exports), not committed
                         (~165MB) -- see Data below
data/raw/genomics/       Small hand-crafted VCF (5 samples, chr21), see Data below
data/raw/genomics_snpeff_db/
                         SnpEff's GRCh37.75 annotation database, trimmed to chr21, not committed
                         (~106MB) -- see Data below
data/processed/          Processed data and the local SQLite database (not committed)
docker/                 Dockerfile
tests/                  pytest suite
```

## Data: where to place raw files

`data/raw/emr/` and `data/raw/genomics/` are committed to this repo already (small enough, and
part of what makes the demo runnable out of the box). The other two raw data sources are not
committed and need to be placed manually:

### Wearable data (not committed, ~165MB)

`src/wearable_pipeline/` expects the public Wearable Exam Stress Dataset (Empatica E4 wristband
exports -- ACC/BVP/EDA/HR/IBI/TEMP/tags CSVs, published on PhysioNet) at:

```
data/raw/wearable/<subject>/<session>/
```

One subfolder per subject (e.g. `S1`-`S5`) and per exam session (`Final`, `Midterm 1`,
`Midterm 2`), filenames unchanged from the distribution. Download it from PhysioNet and place it
at that path to reproduce the demo.

### Genomics data

`src/genomics_pipeline/` expects a single VCF (`.vcf` or `.vcf.gz`) at `data/raw/genomics/`,
subset to 5 sample columns and (for a tractable demo) a single chromosome.
`data/raw/genomics/5patients_test.vcf.gz` is already committed: a small, hand-crafted, valid VCF
(10 variants on chr21) rather than a real 1000 Genomes download -- the public 1000 Genomes FTP
mirrors were unreachable when this phase was built (see PROGRESS.md's Phase 3 notes for what that
means for pipeline output -- pathway enrichment scores come out neutral on this placeholder data).
Swap in a real VCF at that same path to use real variant data instead.

The pipeline also needs SnpEff's GRCh37.75 annotation database at
`data/raw/genomics_snpeff_db/GRCh37.75/`. It isn't committed (~106MB) and SnpEff's own downloader
(`snpEff download GRCh37.75`) pulls from a host that was unreachable in this project's build
environment, so it isn't fetched automatically either. To populate it:

- If you have network access, run `snpEff download GRCh37.75` with any local SnpEff install and
  copy the resulting `data/GRCh37.75/` directory into `data/raw/genomics_snpeff_db/GRCh37.75/`
  (or point `SNPEFF_DATA_DIR`, see `src/genomics_pipeline/config.py`, at wherever you put it).
- Otherwise, obtain `GRCh37.75/snpEffectPredictor.bin` (required) plus `cytoBand.txt.gz` and
  `pwms.bin` and `sequence.<chrom>.bin` for whichever chromosome(s) your VCF covers
  (`sequence.21.bin` for the committed demo VCF) from any existing SnpEff 5.x GRCh37.75 install.

PLINK (`plink1.9`) and SnpEff are resolved via `PATH`, then `PLINK_BIN`/`SNPEFF_BIN` env var
overrides, then a local conda env fallback (see `config.py`) -- already installed in the Docker
image (`docker/Dockerfile`'s `apt-get install plink1.9 snpeff default-jre-headless`); install both
locally too if running the genomics pipeline outside Docker.

## Local development setup (without Docker)

```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # NLP engine Presidio uses for de-identification
```

Run just the API: `uvicorn api.main:app --reload`.
Run just the UI: `API_BASE_URL=http://localhost:8000 streamlit run app/main.py` (that's the
default if `API_BASE_URL` is unset, so you can usually omit it when running both locally).

## API

Read-only access to the digital twin store (Phase 4) and the fusion layer's generated hypotheses
(Phase 5) -- nothing here runs a pipeline or calls the Claude API on request, it only serves
already-computed, already-stored data.

- `GET /patients` -- every patient_id with a stored digital twin
- `GET /patient/{id}/full-twin` -- the whole twin: all three domains, each labeled with its own
  `pipeline_version`
- `GET /patient/{id}/emr`, `/genomic`, `/wearable` -- just one domain's embedding + pipeline_version
- `GET /patient/{id}/hypotheses` -- stored hypotheses citing this patient, each with its
  `source_embedding_ids` (traceability references)

## Database

The database layer uses SQLAlchemy, configured entirely through the `DATABASE_URL` environment
variable (see `.env.example`). If unset, it defaults to a local SQLite file at
`data/processed/twin.db`.

### Migrating to PostgreSQL later

No code changes are required -- only configuration:

1. Uncomment the `postgres` service block (and the `volumes:` block at the bottom) in
   `docker-compose.yml`.
2. Set `DATABASE_URL` in `.env` to a PostgreSQL connection string, e.g.:
   ```
   DATABASE_URL=postgresql+psycopg2://twin:twin@postgres:5432/twin
   ```
   (matching the `postgres` service's `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`, which
   default to `twin`/`twin`/`twin` in the commented-out block).
3. `docker compose up --build`.

`src/db/session.py` reads `DATABASE_URL` at import time and builds the SQLAlchemy engine from
whatever it finds there -- SQLite-specific behavior (the `check_same_thread` connect arg) is only
applied when the URL starts with `sqlite`, so a Postgres URL takes the normal SQLAlchemy/psycopg2
path automatically. Every model in this project is a standard SQLAlchemy table with portable
column types (strings, integers, floats, JSON), so nothing in `src/`, `api/`, or `app/` needs to
change either way.

## Tests

```bash
pytest
```

or inside Docker:

```bash
docker compose run --rm app pytest
```

87 tests across nine test files, covering all seven phases (EMR, wearable, genomics, digital twin
+ orchestration, fusion layer, governance, API) -- pass in both environments. See PROGRESS.md's
Test Status section for the full history and per-phase breakdown.
