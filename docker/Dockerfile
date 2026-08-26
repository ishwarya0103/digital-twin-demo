FROM python:3.11-slim

WORKDIR /app

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
