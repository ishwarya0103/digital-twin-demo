# Digital Twin Demo

A small (2-5 patient) working demo of the light digital twin architecture described in
*A Scalable Generative AI Architecture for Lightweight Digital Twins in Precision Medicine and Drug
Discovery* and its accompanying `Full_Architecture_and_Tech_Stack.pdf` reference.

The architecture integrates three independent, patient-level data domains -- EMR, genomics, and
wearables -- into structured embeddings, aggregates them into a "light digital twin" per patient, and
reasons across them with a generative semantic fusion layer to produce traceable, auditable research
hypotheses. It is retrospective and advisory only: no real-time simulation, no autonomous clinical
decisions.

See [PROGRESS.md](PROGRESS.md) for current phase status.

## Project layout

```
src/emr_pipeline/       EMR extraction -> clinical state vector
src/wearable_pipeline/  Wearable time-series -> physiological state profile ("ouch meter")
src/genomics_pipeline/  GWAS + pathway aggregation -> genomic pathway profile
src/digital_twin/       Digital Twin Abstraction Layer (combines the three embeddings)
src/fusion_layer/       Generative Semantic Fusion Layer
src/governance/         De-identification, audit logging, institutional boundaries
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
                         (~165MB) -- see Wearable data below
data/raw/genomics/       Small hand-crafted VCF (5 samples, chr21), see Genomics data below
data/raw/genomics_snpeff_db/
                         SnpEff's GRCh37.75 annotation database, trimmed to chr21, not committed
                         (~106MB) -- see Genomics data below
data/processed/          Processed data and the local SQLite database (not committed)
docker/                 Dockerfile
tests/                  pytest suite
```

## Setup

```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # NLP engine Presidio uses for de-identification
```

## Running with Docker

```bash
docker compose up --build
```

The app is served on `http://localhost:8000`. `./data` is mounted into the container so the SQLite
database at `data/processed/twin.db` persists across restarts.

## Wearable data

`src/wearable_pipeline/` expects the public Wearable Exam Stress Dataset (Empatica E4 wristband
exports -- ACC/BVP/EDA/HR/IBI/TEMP/tags CSVs, published on PhysioNet) at
`data/raw/wearable/<subject>/<session>/`, one subfolder per subject (e.g. `S1`-`S5`) and per exam
session (`Final`, `Midterm 1`, `Midterm 2`), filenames unchanged from the distribution. It isn't
committed here (~165MB of raw sensor CSVs) -- download it from PhysioNet and place it at that path
to reproduce the demo.

## Genomics data

`src/genomics_pipeline/` expects a single VCF (`.vcf` or `.vcf.gz`) at `data/raw/genomics/`,
subset to the same 5 sample IDs used elsewhere and (for a tractable demo) a single chromosome.
`data/raw/genomics/5patients_test.vcf.gz` is committed here: a small, hand-crafted, valid VCF (10
variants on chr21) rather than a real 1000 Genomes download -- the public 1000 Genomes FTP mirrors
were unreachable when this phase was built. See PROGRESS.md's Phase 3 notes for what that means for
pipeline output (pathway enrichment scores come out neutral).

The pipeline also needs SnpEff's GRCh37.75 annotation database at
`data/raw/genomics_snpeff_db/GRCh37.75/`. It isn't committed (~106MB) and SnpEff's own downloader
(`snpEff download GRCh37.75`) pulls from `snpeff.blob.core.windows.net`, which was also unreachable
in this project's dev/build environment -- so it isn't fetched automatically at Docker build time
either (see `docker/Dockerfile`'s comment on this). To reproduce:

- If you have network access to that host, run `snpEff download GRCh37.75` with any local SnpEff
  install and point `SNPEFF_DATA_DIR` (see `src/genomics_pipeline/config.py`) at the resulting
  `data/GRCh37.75/` directory, or copy it into `data/raw/genomics_snpeff_db/GRCh37.75/`.
- Otherwise, obtain `GRCh37.75/snpEffectPredictor.bin` (required) plus `cytoBand.txt.gz` and
  `pwms.bin` (used for effect prediction) and `sequence.<chrom>.bin` for whichever chromosome(s)
  your VCF covers (`sequence.21.bin` for the committed demo VCF) from any existing SnpEff 5.x
  GRCh37.75 install, and place them under `data/raw/genomics_snpeff_db/GRCh37.75/`.

PLINK (`plink1.9`) and SnpEff are resolved via `PATH`, then `PLINK_BIN`/`SNPEFF_BIN` env var
overrides, then a local conda env fallback (see `config.py`) -- installed via
`docker/Dockerfile`'s `apt-get install plink1.9 snpeff default-jre-headless` for the Docker image,
or install both locally (e.g. via a `genomics` conda env) for the local venv.

## Database

The database layer uses SQLAlchemy, configured entirely through the `DATABASE_URL` environment
variable (see `.env.example`). If unset, it defaults to a local SQLite file at
`data/processed/twin.db`. Setting `DATABASE_URL` to a PostgreSQL connection string later requires no
code changes.

## Tests

```bash
pytest
```

or inside Docker:

```bash
docker compose run --rm app pytest
```
