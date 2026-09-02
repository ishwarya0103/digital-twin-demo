"""Clusters patients by digital twin embedding similarity (architecture doc Section 5,
"cross-modal pattern detection": "Identifies subpatient clusters sharing medication-response
trajectories and pathway signatures ... Embedding similarity/clustering (e.g., cosine
similarity, k-means/HDBSCAN) feeding the LLM's reasoning context"). Grouping patients before
hypothesis generation lets each generated hypothesis describe a cohort subgroup rather than a
single patient in isolation.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.digital_twin.models import DigitalTwin
from src.digital_twin.retrieval import DOMAINS


def _concatenated_embedding(twin: DigitalTwin) -> np.ndarray:
    """The digital twin's own definition (architecture doc Section 4): the concatenation of
    the three domain embeddings, in a fixed domain order."""
    return np.concatenate([np.asarray(getattr(twin, f"{d}_embedding"), dtype=np.float64) for d in DOMAINS])


def cluster_twins(
    twins: list[DigitalTwin], n_clusters: int = 2, random_state: int = 0
) -> dict[int, list[DigitalTwin]]:
    """K-means over each patient's concatenated (EMR + genomic + wearable) embedding,
    per-feature standardized first so no domain dominates purely because it has more
    dimensions or a larger natural scale (EMR's 768 dims would otherwise swamp genomic's ~15
    and wearable's 16 in a raw Euclidean distance). Returns {cluster_label: [twin, ...]}.

    `n_clusters` is capped at the number of twins given -- k-means requires n_clusters <=
    n_samples, and with 1 twin or 1 requested cluster everything trivially falls into a single
    cluster without needing to fit a model."""
    if not twins:
        return {}

    n_clusters = max(1, min(n_clusters, len(twins)))
    matrix = np.stack([_concatenated_embedding(t) for t in twins])

    if n_clusters == 1:
        labels = np.zeros(len(twins), dtype=int)
    else:
        scaled = StandardScaler().fit_transform(matrix)
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(scaled)

    clusters: dict[int, list[DigitalTwin]] = {}
    for label, twin in zip(labels, twins):
        clusters.setdefault(int(label), []).append(twin)
    return clusters
