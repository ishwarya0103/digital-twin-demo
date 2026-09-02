"""Orchestrates the Generative Semantic Fusion Layer end-to-end for a cohort of digital twins
(architecture doc Section 5): cluster patients by embedding similarity, upsert every twin's
embeddings into chromadb, format each cluster into structured text, then generate and store one
traceable hypothesis per cluster.
"""

import anthropic
import chromadb
from sqlalchemy.orm import Session

from src.digital_twin.models import DigitalTwin
from src.fusion_layer.clustering import cluster_twins
from src.fusion_layer.formatting import format_cluster_summary
from src.fusion_layer.models import Hypothesis
from src.fusion_layer.reasoning import DEFAULT_MODEL, generate_and_store_hypothesis
from src.fusion_layer.vector_store import upsert_twin_embeddings


def run_fusion_layer_for_cohort(
    db: Session,
    chroma_client: chromadb.api.ClientAPI,
    anthropic_client: anthropic.Anthropic,
    twins: list[DigitalTwin],
    n_clusters: int = 2,
    model: str = DEFAULT_MODEL,
) -> list[Hypothesis]:
    """One hypothesis per cluster. Every twin's embeddings are upserted into chromadb before
    generation, so a hypothesis's source_embedding_ids always name entries that already exist
    in the vector store, not just IDs computed on the fly."""
    clusters = cluster_twins(twins, n_clusters=n_clusters)

    hypotheses = []
    for cluster_id, cluster_members in clusters.items():
        candidate_embedding_ids = []
        for twin in cluster_members:
            ids_by_domain = upsert_twin_embeddings(chroma_client, twin)
            candidate_embedding_ids.extend(ids_by_domain.values())

        summary_text = format_cluster_summary(cluster_members, cluster_id)
        hypothesis = generate_and_store_hypothesis(
            db, anthropic_client, summary_text, candidate_embedding_ids, model=model
        )
        hypotheses.append(hypothesis)

    return hypotheses
