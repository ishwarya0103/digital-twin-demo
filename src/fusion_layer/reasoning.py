"""Layer 3 -- Generative Semantic Fusion Layer (architecture doc Section 5): "Prompt-constrained
reasoning engine ... Generates hypotheses across clinical, genomic, and physiological
dimensions, bound to schema templates rather than free text" via a "Generative LLM ... accessed
via API, with engineered system prompts and output schemas."

The model reasons only over the structured cluster summary text produced by formatting.py --
never raw EMR text, genomic sequences, or physiological signals, none of which this module ever
has access to. Output is forced into HYPOTHESIS_JSON_SCHEMA via Claude's tool-use API
(`tool_choice` pinned to a single tool), matching the doc's "Structured output enforcement"
component ("JSON-schema-constrained generation / function-calling"), and every hypothesis must
cite at least one of the specific embedding IDs it was given as evidence -- the "Traceability
layer" component ("Links every generated hypothesis back to the specific embedding evidence
that produced it").

Per the architecture doc's Section 5 note: this layer is a partial, research-stage workflow-
automation engine for hypothesis generation, not a production clinical system -- output is
advisory only, never a diagnosis, prediction, or treatment recommendation.
"""

import os

import anthropic
import jsonschema
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.fusion_layer.models import Hypothesis

DEFAULT_MODEL = "claude-sonnet-5"

HYPOTHESIS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "subgroup_trait": {
            "type": "string",
            "minLength": 1,
            "description": "A short, specific description of the trait/pattern this patient "
            "subgroup shares across domains.",
        },
        "supporting_evidence": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Specific observations, drawn only from the provided structured "
            "summary, that support this hypothesis.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Self-assessed confidence in this hypothesis.",
        },
        "source_embedding_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "Which of the provided candidate embedding IDs this hypothesis is "
            "grounded in.",
        },
    },
    "required": ["subgroup_trait", "supporting_evidence", "confidence", "source_embedding_ids"],
    "additionalProperties": False,
}

HYPOTHESIS_TOOL = {
    "name": "record_hypothesis",
    "description": (
        "Records one structured, traceable research hypothesis about a patient subgroup, "
        "grounded only in the structured embedding summary provided in this conversation -- "
        "never in outside medical knowledge about specific real patients."
    ),
    "input_schema": HYPOTHESIS_JSON_SCHEMA,
}

_SYSTEM_PROMPT = (
    "You are the generative semantic fusion layer of a retrospective, advisory-only clinical "
    "research tool. You reason ONLY over the structured, already-de-identified embedding "
    "summary statistics given to you in the user message -- you have no access to, and must "
    "never assume or invent, raw clinical notes, genomic sequences, or physiological signals. "
    "Your output is a hypothesis for human researchers to investigate further, not a "
    "diagnosis, prediction, or treatment recommendation. Call the record_hypothesis tool "
    "exactly once with your hypothesis."
)

# Transient failures worth retrying; authentication/validation/quota errors are not (retrying
# a bad API key or a malformed request just fails the same way N more times).
_RETRYABLE_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


def get_anthropic_client() -> anthropic.Anthropic:
    """Reads ANTHROPIC_API_KEY from the environment (via .env, see .env.example) -- never
    hardcoded. max_retries=0 because `_call_claude` below owns the retry/backoff policy."""
    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example) -- never hardcode "
            "it in code or commit it."
        )
    return anthropic.Anthropic(api_key=api_key, max_retries=0)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _call_claude(client: anthropic.Anthropic, **kwargs):
    return client.messages.create(**kwargs)


def _build_prompt(cluster_summary_text: str, candidate_source_embedding_ids: list[str]) -> str:
    ids_list = "\n".join(f"  - {eid}" for eid in candidate_source_embedding_ids)
    return (
        "Structured cluster summary (embedding-level statistics only, no raw patient data):\n\n"
        f"{cluster_summary_text}\n\n"
        "Candidate source_embedding_ids you may cite in your response (cite only IDs from this "
        "list, and cite at least one):\n"
        f"{ids_list}\n\n"
        "Generate one cross-modal research hypothesis about what this subgroup's clinical, "
        "genomic, and wearable-derived embedding statistics might have in common, using the "
        "record_hypothesis tool."
    )


def generate_hypothesis(
    client: anthropic.Anthropic,
    cluster_summary_text: str,
    candidate_source_embedding_ids: list[str],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> dict:
    """Calls Claude, forced (via `tool_choice`) to respond with exactly one `record_hypothesis`
    tool call, and returns its validated `input` dict. Raises `jsonschema.ValidationError` if
    the response doesn't match HYPOTHESIS_JSON_SCHEMA, and `ValueError` if it cites a
    source_embedding_id outside `candidate_source_embedding_ids` -- a hallucinated ID would
    break the traceability guarantee this whole layer exists to provide."""
    if not candidate_source_embedding_ids:
        raise ValueError("candidate_source_embedding_ids must be non-empty")

    response = _call_claude(
        client,
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        tools=[HYPOTHESIS_TOOL],
        tool_choice={"type": "tool", "name": HYPOTHESIS_TOOL["name"]},
        messages=[
            {"role": "user", "content": _build_prompt(cluster_summary_text, candidate_source_embedding_ids)}
        ],
    )

    tool_use = next((block for block in response.content if getattr(block, "type", None) == "tool_use"), None)
    if tool_use is None:
        raise ValueError("Claude response did not include a record_hypothesis tool call")

    hypothesis = tool_use.input
    jsonschema.validate(instance=hypothesis, schema=HYPOTHESIS_JSON_SCHEMA)

    unknown_ids = set(hypothesis["source_embedding_ids"]) - set(candidate_source_embedding_ids)
    if unknown_ids:
        raise ValueError(f"Hypothesis cited source_embedding_ids outside the provided candidates: {unknown_ids}")

    return hypothesis


def store_hypothesis(db: Session, hypothesis: dict, model: str) -> Hypothesis:
    """Persists an already-generated hypothesis dict. Re-validates against
    HYPOTHESIS_JSON_SCHEMA defensively -- this function shouldn't trust that every caller
    already validated its input."""
    jsonschema.validate(instance=hypothesis, schema=HYPOTHESIS_JSON_SCHEMA)

    row = Hypothesis(
        model=model,
        subgroup_trait=hypothesis["subgroup_trait"],
        supporting_evidence=hypothesis["supporting_evidence"],
        confidence=hypothesis["confidence"],
        source_embedding_ids=hypothesis["source_embedding_ids"],
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def generate_and_store_hypothesis(
    db: Session,
    client: anthropic.Anthropic,
    cluster_summary_text: str,
    candidate_source_embedding_ids: list[str],
    model: str = DEFAULT_MODEL,
) -> Hypothesis:
    hypothesis = generate_hypothesis(client, cluster_summary_text, candidate_source_embedding_ids, model=model)
    return store_hypothesis(db, hypothesis, model=model)
