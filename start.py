"""
start.py — Guaranteed startup script. Run this instead of uvicorn directly.
Usage: python start.py
"""
import os
import sys
import secrets
from pathlib import Path

# ── Step 1: Load .env file ─────────────────────────────────────────────────
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
                    os.environ[key] = value
else:
    print("WARNING: .env not found — using defaults")

# ── Step 2: Validate / generate vault key ─────────────────────────────────
vault_key = os.environ.get("PHI_VAULT_KEY", "")

if not vault_key or vault_key == "your-32-byte-hex-key-here":
    # Auto-generate for this session
    vault_key = secrets.token_hex(32)
    os.environ["PHI_VAULT_KEY"] = vault_key
    print(f"Auto-generated PHI_VAULT_KEY for this session.")
    print(f"Add this to your .env to make it permanent:")
    print(f"PHI_VAULT_KEY={vault_key}")
elif len(vault_key) != 64:
    print(f"WARNING: PHI_VAULT_KEY is {len(vault_key)} chars, expected 64. Auto-generating.")
    vault_key = secrets.token_hex(32)
    os.environ["PHI_VAULT_KEY"] = vault_key

# ── Step 3: Set defaults for any missing vars ─────────────────────────────
defaults = {
    "VAULT_BACKEND":          "sqlite",
    "PHI_VAULT_SQLITE_PATH":  "./vault.sqlite",
    "AUDIT_BACKEND":          "sqlite",
    "AUDIT_SQLITE_PATH":      "./audit.sqlite",
    "LLM_PROVIDER":           "stub",
    "RAG_BACKEND":            "local",
    "API_HOST":               "0.0.0.0",
    "API_PORT":               "8000",
}
for k, v in defaults.items():
    if not os.environ.get(k):
        os.environ[k] = v

# ── Step 4: Print status ──────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=== Starting Healthcare Claims Agent ===")
    print(f"  LLM Provider : {os.environ['LLM_PROVIDER']}")
    print(f"  Vault        : {os.environ['VAULT_BACKEND']}")
    print(f"  Audit        : {os.environ['AUDIT_BACKEND']}")
    print(f"  RAG          : {os.environ['RAG_BACKEND']}")
    print(f"  Vault key    : SET ({len(os.environ['PHI_VAULT_KEY'])} chars)")
    print()
    print("  Swagger UI   : http://localhost:8000/docs")
    print("  API docs     : http://localhost:8000/redoc")
    print()

    # ── Step 5: Start server ──────────────────────────────────────────────────
    import uvicorn
    uvicorn.run(
        "agent.api:app",
        host   = os.environ.get("API_HOST", "0.0.0.0"),
        port   = int(os.environ.get("API_PORT", "8000")),
        reload = False,
    )
