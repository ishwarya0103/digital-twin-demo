# CLAUDE.md

## Purpose

This project is a small (2-5 patient) working demo of the light digital twin architecture described
in the attached RWE journal paper and `Full_Architecture_and_Tech_Stack.pdf`. It integrates EMR,
genomic, and wearable data into per-patient structured embeddings, combines them into a "digital
twin" abstraction, and (in later phases) reasons across them with a generative semantic fusion layer
to produce traceable research hypotheses. The system is retrospective and advisory only -- it does
not perform real-time clinical decision-making.

## Tech stack

- Python
- FastAPI (`api/`) -- backend API
- Streamlit (`app/`) -- UI
- SQLAlchemy (`src/db/`) -- database layer, connection string from `DATABASE_URL`, defaults to a
  local SQLite file (`data/processed/twin.db`); swapping to PostgreSQL later requires only an env
  var change, no code changes
- Docker / Docker Compose -- containerized app, `./data` mounted as a volume so SQLite persists
  outside the container; a commented-out `postgres` service in `docker-compose.yml` is a placeholder
  for later migration
- chromadb -- included in `requirements.txt` for later use as the embedding/vector store, not yet
  wired up
- pytest -- test suite (`tests/`)
- Presidio, transformers, torch, numpy, scipy, neurokit2, gseapy, anthropic -- pipeline-specific
  dependencies for later phases (EMR NLP, wearable signal processing, genomics pathway analysis,
  generative fusion)

## Current status

See [PROGRESS.md](PROGRESS.md) for current phase, completed work, test status, and next steps.
