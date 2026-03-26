"""
start.py — Startup script for Windows.
Place in: healthcare-claims-agent/start.py
Run with: python start.py
"""
import os
import sys
import secrets
from pathlib import Path

# ── Step 1: Load .env manually ────────────────────────────────────────────
# Look for .env in same folder as this script
env_path = Path(__file__).parent / ".env"

if env_path.exists():
    print(f"Loading .env from: {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key   = key.strip()
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ[key] = value
else:
    print(f"WARNING: .env not found at {env_path}")

# ── Step 2: Validate vault key ────────────────────────────────────────────
vault_key = os.environ.get("PHI_VAULT_KEY", "")

if not vault_key or vault_key == "your-32-byte-hex-key-here":
    vault_key = secrets.token_hex(32)
    os.environ["PHI_VAULT_KEY"] = vault_key
    print(f"\nAuto-generated PHI_VAULT_KEY.")
    print(f"Add to .env to keep it permanent:\nPHI_VAULT_KEY={vault_key}\n")
elif len(vault_key) != 64:
    vault_key = secrets.token_hex(32)
    os.environ["PHI_VAULT_KEY"] = vault_key
    print(f"Key was wrong length — auto-generated new one.")

# ── Step 3: Set defaults ──────────────────────────────────────────────────
defaults = {
    "VAULT_BACKEND":         "sqlite",
    "PHI_VAULT_SQLITE_PATH": "./vault.sqlite",
    "AUDIT_BACKEND":         "sqlite",
    "AUDIT_SQLITE_PATH":     "./audit.sqlite",
    "LLM_PROVIDER":          "stub",
    "RAG_BACKEND":           "local",
}
for k, v in defaults.items():
    if not os.environ.get(k):
        os.environ[k] = v

# ── Step 4: Print status ──────────────────────────────────────────────────
print()
print("=" * 45)
print("  Healthcare Claims Agent — Starting")
print("=" * 45)
print(f"  LLM Provider : {os.environ['LLM_PROVIDER']}")
print(f"  Vault        : {os.environ['VAULT_BACKEND']}")
print(f"  Vault key    : SET ({len(os.environ['PHI_VAULT_KEY'])} chars)")
print(f"  Swagger UI   : http://localhost:8000/docs")
print("=" * 45)
print()

# ── Step 5: Start server (no reload — fixes Windows multiprocessing error) 
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host   = "0.0.0.0",
        port   = 8000,
        reload = False,   # reload=True breaks on Windows — use False
    )