"""Pydantic response models for api/main.py. Kept separate from the route definitions so the
response *shape* (what a client can rely on) is easy to read without the routing/DB-lookup
logic around it.
"""

from datetime import datetime

from pydantic import BaseModel


class DomainEmbedding(BaseModel):
    pipeline_version: str
    embedding: list[float]


class FullTwinResponse(BaseModel):
    """GET /patient/{id}/full-twin -- the whole digital twin: all three domains, each still
    clearly labeled with its own pipeline_version (architecture doc Section 4 traceability
    requirement), nested under its own key so a domain-filtered endpoint's response (below)
    is a strict subset of this one's shape, never a differently-shaped ad hoc object."""

    patient_id: str
    version: int
    created_at: datetime
    emr: DomainEmbedding
    genomic: DomainEmbedding
    wearable: DomainEmbedding


class DomainResponse(BaseModel):
    """GET /patient/{id}/{emr,genomic,wearable} -- exactly one domain's slice. Deliberately
    has no field that could carry another domain's data -- there's no way to construct this
    model with EMR/genomic/wearable data mixed together."""

    patient_id: str
    version: int
    domain: str
    pipeline_version: str
    embedding: list[float]


class HypothesisResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    created_at: datetime
    model: str
    subgroup_trait: str
    supporting_evidence: list[str]
    confidence: float
    source_embedding_ids: list[str]


class HypothesesResponse(BaseModel):
    patient_id: str
    hypotheses: list[HypothesisResponse]


class PatientsResponse(BaseModel):
    patient_ids: list[str]
