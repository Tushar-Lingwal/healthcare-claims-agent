FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/build_code_db.py
RUN python scripts/build_rag_db.py
RUN mkdir -p data/vault data/audit

ENV VAULT_BACKEND=sqlite \
    AUDIT_BACKEND=sqlite \
    PHI_VAULT_SQLITE_PATH=/app/data/vault/vault.sqlite \
    AUDIT_SQLITE_PATH=/app/data/audit/audit.sqlite \
    RAG_BACKEND=local \
    LLM_PROVIDER=rules \
    AUTH_ENABLED=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# NO reload flag — causes multiprocessing crash on Linux containers
# NO start.py — use uvicorn directly, cleanly
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]