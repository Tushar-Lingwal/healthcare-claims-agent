"""
start.py — Local development launcher only.
For production/Railway: the Dockerfile runs uvicorn directly.
"""
import os
import sys
import secrets
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    print(f"Loading .env from: {env_path.absolute()}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ.setdefault(key, value)
else:
    print("WARNING: .env not found — using defaults")

# ── Generate vault key if missing ────────────────────────────────
vault_key = os.environ.get("PHI_VAULT_KEY", "")
if not vault_key or len(vault_key) != 64:
    vault_key = secrets.token_hex(32)
    os.environ["PHI_VAULT_KEY"] = vault_key
    print(f"Auto-generated PHI_VAULT_KEY for this session.")
    print(f"Add to .env:  PHI_VAULT_KEY={vault_key}")

# ── Defaults ──────────────────────────────────────────────────────
for k, v in {
    "VAULT_BACKEND":         "sqlite",
    "PHI_VAULT_SQLITE_PATH": "./vault.sqlite",
    "AUDIT_BACKEND":         "sqlite",
    "AUDIT_SQLITE_PATH":     "./audit.sqlite",
    "LLM_PROVIDER":          "rules",
    "RAG_BACKEND":           "local",
    "API_HOST":              "0.0.0.0",
    "API_PORT":              "8000",
}.items():
    os.environ.setdefault(k, v)

# ── Status ────────────────────────────────────────────────────────
print()
print("=== Starting Healthcare Claims Agent ===")
print(f"  LLM Provider : {os.environ['LLM_PROVIDER']}")
print(f"  Vault        : {os.environ['VAULT_BACKEND']}")
print(f"  Audit        : {os.environ['AUDIT_BACKEND']}")
print(f"  RAG          : {os.environ['RAG_BACKEND']}")
print(f"  Vault key    : SET ({len(os.environ['PHI_VAULT_KEY'])} chars)")
print()
print(f"  Swagger UI   : http://localhost:{os.environ['API_PORT']}/docs")
print(f"  API docs     : http://localhost:{os.environ['API_PORT']}/redoc")
print()

# ── Run — NO reload, NO multiprocessing ──────────────────────────
import uvicorn
uvicorn.run(
    "agent.api:app",
    host    = os.environ.get("API_HOST", "0.0.0.0"),
    port    = int(os.environ.get("API_PORT", "8000")),
    reload  = False,   # NEVER True — breaks Windows & Linux containers
    workers = 1,
)