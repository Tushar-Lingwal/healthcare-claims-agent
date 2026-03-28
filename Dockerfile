FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build databases during image build
RUN python scripts/build_code_db.py && \
    python scripts/build_rag_db.py && \
    mkdir -p data/vault data/audit

# Runtime env defaults (Railway injects its own PORT + your Variables)
ENV VAULT_BACKEND=sqlite \
    AUDIT_BACKEND=sqlite \
    PHI_VAULT_SQLITE_PATH=/app/data/vault/vault.sqlite \
    AUDIT_SQLITE_PATH=/app/data/audit/audit.sqlite \
    RAG_BACKEND=local \
    LLM_PROVIDER=rules \
    AUTH_ENABLED=false

EXPOSE 8000

# Use shell form so $PORT is expanded by Railway at runtime
CMD uvicorn agent.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log