FROM python:3.11-slim

WORKDIR /app

# PLINK 1.9 and SnpEff (genomics_pipeline) -- SnpEff pulls in default-jre-headless as a dependency.
# The GRCh37.75 SnpEff genome database itself is NOT fetched here (its usual source,
# snpeff.blob.core.windows.net, is unreachable in this project's dev/build environment); it's
# supplied at runtime via the same ./data:/app/data volume mount as the rest of data/raw/, see
# README.md's "Genomics data" section and src/genomics_pipeline/config.py's SNPEFF_DATA_DIR.
RUN apt-get update \
    && apt-get install -y --no-install-recommends plink1.9 snpeff default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
ENV PLINK_BIN=/usr/bin/plink1.9
ENV SNPEFF_BIN=/usr/bin/snpEff

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg
# Pre-fetch and cache Bio_ClinicalBERT into the image so containers don't re-download it on
# every run. Kept before COPY . . (and the model name kept as a literal, matching
# src/emr_pipeline/nlp_extraction.py's MODEL_NAME) so this layer only invalidates on a
# requirements.txt change, not on every source edit.
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')"

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
