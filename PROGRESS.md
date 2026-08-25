# Progress

## Current Phase

Phase 0 -- Project scaffolding

## Status

Complete

## Completed

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
- [x] `requirements.txt` covering current and future-phase dependencies: fastapi, streamlit,
      pytest, presidio, transformers, torch, numpy, scipy, neurokit2, gseapy, anthropic,
      python-dotenv, sqlalchemy, chromadb
- [x] `tests/test_app_startup.py` -- confirms the app starts and connects to the database
- [x] `README.md`, `CLAUDE.md`, `.env.example`, `.gitignore`, `.dockerignore`

## Test Status

`tests/test_app_startup.py::test_app_starts_and_connects_to_db` -- **PASSED** (run inside Docker via
`docker compose run --rm app pytest -v`, 2026-08-25).

## Known Issues / Blockers

- Docker Desktop must be running before `docker compose build`/`run` -- confirm before invoking.
- No pipeline logic yet: `src/emr_pipeline/`, `src/wearable_pipeline/`, `src/genomics_pipeline/`,
  `src/digital_twin/`, `src/fusion_layer/`, `src/governance/` are empty packages awaiting Phase 1+.
- `chromadb` is installed but not yet wired into any code path.

## Next Steps

- Phase 1: EMR pipeline -- FHIR-normalized structured fields, Presidio de-identification, clinical
  NLP extraction (ClinicalBERT / fine-tuned LLM) producing the clinical state vector
- Phase 1: Wearable pipeline -- signal preprocessing, NeuroKit2 feature engineering, time-series
  model producing the physiological state profile ("ouch meter")
- Phase 1: Genomics pipeline -- variant annotation, GWAS, GSEA pathway aggregation producing the
  genomic pathway profile
- Phase 2: Digital Twin Abstraction Layer combining the three embeddings per patient
- Phase 3: Generative Semantic Fusion Layer (Anthropic API, structured/schema-constrained output)
- Seed 2-5 synthetic patients across data/raw/ for demo purposes
