"""Streamlit demo UI: a doctor-facing view over the digital twin store and the fusion layer's
generated hypotheses. Talks to the FastAPI backend (api/) over HTTP -- never imports src/
pipeline code or queries the database directly -- so this UI only ever sees what the API
actually serves, the same contract any other client of the API would have.

Retrospective, advisory-only research tool. Not for real-time clinical decision-making.
"""

import os

import requests
import streamlit as st


def _resolve_api_base_url() -> str:
    """API_BASE_URL is a full URL locally and in docker-compose.yml (e.g.
    "http://app:8000"). On Render, the equivalent env var is instead set via a
    `fromService`/`property: hostport` reference (render.yaml), which returns a bare
    "host:port" with no scheme -- add one so both conventions work unchanged."""
    value = os.environ.get("API_BASE_URL", "http://localhost:8000")
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value


API_BASE_URL = _resolve_api_base_url()

st.set_page_config(page_title="Digital Twin Demo", layout="wide")
st.title("Digital Twin Demo")
st.caption(
    "Retrospective, advisory-only research tool -- for hypothesis generation and cohort "
    "review, not real-time clinical decision-making."
)


def _get(path: str) -> dict | None:
    """GETs `path` from the API. Returns None for a 404 (a valid, expected "nothing here yet"
    response for this demo's endpoints), raises for anything else."""
    response = requests.get(f"{API_BASE_URL}{path}", timeout=10)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def fetch_patient_ids() -> list[str]:
    return _get("/patients")["patient_ids"]


try:
    patient_ids = fetch_patient_ids()
except requests.RequestException as exc:
    st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
    st.stop()

if not patient_ids:
    st.warning("No digital twins found yet -- run the Phase 1-4 pipelines first.")
    st.stop()

patient_id = st.selectbox("Select a patient", patient_ids)

twin_view, hypotheses_view = st.tabs(["Digital Twin", "Hypotheses"])


def _render_emr_summary(summary: dict) -> None:
    any_shown = False
    for key, title in (("diagnoses", "Diagnoses"), ("medications", "Medications"), ("symptoms", "Symptoms")):
        values = summary.get(key) or []
        if values:
            st.markdown(f"**{title}:** {', '.join(values)}")
            any_shown = True
    if not any_shown:
        st.caption("No extracted diagnoses, medications, or symptoms.")


def _render_genomic_summary(summary: dict) -> None:
    pathway_scores = summary.get("pathway_scores") or {}
    elevated = sorted(((n, s) for n, s in pathway_scores.items() if s), key=lambda kv: -abs(kv[1]))
    if elevated:
        st.markdown("**Elevated pathways:**")
        for name, score in elevated:
            st.markdown(f"- {name}: {score:+.2f}")
    else:
        st.caption("No elevated genomic pathway signal.")


def _render_wearable_summary(summary: dict) -> None:
    interpretation = summary.get("interpretation", "No wearable activation data available.")
    mean_score = summary.get("mean_activation_score")
    st.markdown(f"**{interpretation.capitalize()}**")
    if mean_score is not None:
        st.caption(f"mean activation score: {mean_score:.2f} across {summary.get('num_windows', 0)} windows")


_SUMMARY_RENDERERS = {
    "emr": _render_emr_summary,
    "genomic": _render_genomic_summary,
    "wearable": _render_wearable_summary,
}


def _render_domain(domain_key: str, label: str, domain: dict) -> None:
    """Plain-language clinical summary as the primary view; the raw embedding vector (what
    actually feeds clustering/similarity search) tucked into an expander for anyone who wants
    the underlying numbers rather than the interpretation."""
    st.markdown(f"**{label}**")
    st.caption(f"pipeline_version: `{domain['pipeline_version']}`")
    _SUMMARY_RENDERERS[domain_key](domain.get("summary") or {})

    with st.expander("Technical detail (raw embedding vector)"):
        st.caption(f"dimension: {len(domain['embedding'])}")
        st.line_chart(domain["embedding"])


with twin_view:
    twin = _get(f"/patient/{patient_id}/full-twin")
    if twin is None:
        st.info(f"No digital twin has been assembled for {patient_id} yet.")
    else:
        st.subheader(f"{patient_id} -- twin version {twin['version']}")
        st.caption(f"Assembled at {twin['created_at']}")

        all_tab, emr_tab, genomic_tab, wearable_tab = st.tabs(
            ["All domains", "EMR only", "Genomic only", "Wearable only"]
        )
        with all_tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                _render_domain("emr", "EMR / Clinical Summary", twin["emr"])
            with col2:
                _render_domain("genomic", "Genomic Pathway Profile", twin["genomic"])
            with col3:
                _render_domain("wearable", "Wearable Physiological Profile", twin["wearable"])
        with emr_tab:
            _render_domain("emr", "EMR / Clinical Summary", twin["emr"])
        with genomic_tab:
            _render_domain("genomic", "Genomic Pathway Profile", twin["genomic"])
        with wearable_tab:
            _render_domain("wearable", "Wearable Physiological Profile", twin["wearable"])

with hypotheses_view:
    data = _get(f"/patient/{patient_id}/hypotheses")
    hypotheses = (data or {}).get("hypotheses", [])

    if not hypotheses:
        st.info(f"No hypotheses have been generated for {patient_id} yet.")

    for hypothesis in hypotheses:
        with st.container(border=True):
            st.markdown(f"#### {hypothesis['subgroup_trait']}")
            st.caption(
                f"confidence: {hypothesis['confidence']:.2f} -- model: {hypothesis['model']} "
                f"-- generated: {hypothesis['created_at']}"
            )

            st.markdown("**Supporting evidence**")
            for evidence in hypothesis["supporting_evidence"]:
                st.markdown(f"- {evidence}")

            st.markdown("**Source embeddings (traceability)**")
            st.code("\n".join(hypothesis["source_embedding_ids"]), language=None)
