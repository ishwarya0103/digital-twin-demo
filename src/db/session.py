import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Default location for the local SQLite file when DATABASE_URL is unset.
# Switching to Postgres later only requires setting DATABASE_URL in .env --
# no code changes.
_DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "twin.db"
_DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL") or f"sqlite:///{_DEFAULT_SQLITE_PATH}"

# SQLite needs this flag for use across threads (e.g. FastAPI's threadpool);
# other backends ignore it.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
