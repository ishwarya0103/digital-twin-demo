"""Stores each digital twin's domain embeddings in chromadb, so similarity search over
patients' embeddings is available going forward -- independent of, and in addition to, the SQL
row in DigitalTwin (src/digital_twin/models.py), which remains the system of record. Matches
the architecture doc's Section 9 tech-stack line for the Digital Twin Layer: "Storage |
Versioned feature store / vector store".

One collection per domain rather than one shared collection: EMR (768-dim), genomic (~15-dim),
and wearable (16-dim) embeddings have different dimensionality, and a chromadb collection
expects uniform embedding dimensionality across all of its entries.
"""

from pathlib import Path

import chromadb

from src.digital_twin.models import DigitalTwin
from src.digital_twin.retrieval import DOMAINS
from src.fusion_layer.formatting import embedding_id

DEFAULT_PERSIST_DIR = "data/processed/chroma_db"


def get_chroma_client(persist_directory: str = DEFAULT_PERSIST_DIR) -> chromadb.api.ClientAPI:
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)


def _collection_name(domain: str) -> str:
    return f"digital_twin_{domain}"


def upsert_twin_embeddings(client: chromadb.api.ClientAPI, twin: DigitalTwin) -> dict[str, str]:
    """Upserts all three of a twin's domain embeddings into their respective per-domain
    collections (keyed by `embedding_id()`, re-upserting the same twin version is idempotent).
    Returns {domain: embedding_id} for the entries just written -- the same IDs used in
    formatting.py's prompt text and, downstream, in a hypothesis's source_embedding_ids."""
    ids_by_domain = {}
    for domain in DOMAINS:
        collection = client.get_or_create_collection(_collection_name(domain))
        eid = embedding_id(twin.patient_id, twin.version, domain)
        collection.upsert(
            ids=[eid],
            embeddings=[getattr(twin, f"{domain}_embedding")],
            metadatas=[
                {
                    "patient_id": twin.patient_id,
                    "version": twin.version,
                    "domain": domain,
                    "pipeline_version": getattr(twin, f"{domain}_pipeline_version"),
                }
            ],
        )
        ids_by_domain[domain] = eid
    return ids_by_domain


def query_similar(
    client: chromadb.api.ClientAPI, domain: str, embedding: list[float], n_results: int = 5
) -> chromadb.api.types.QueryResult:
    """Nearest-neighbor similarity search within one domain's collection -- the "similarity
    search... available going forward" this module exists to provide."""
    collection = client.get_or_create_collection(_collection_name(domain))
    return collection.query(query_embeddings=[embedding], n_results=n_results)
