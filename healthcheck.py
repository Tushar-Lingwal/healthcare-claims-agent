"""
healthcheck.py — Run this locally to verify the app starts cleanly.
python healthcheck.py
"""
import os, sys, secrets

# Minimal env for import test
os.environ.setdefault("PHI_VAULT_KEY",          secrets.token_hex(32))
os.environ.setdefault("VAULT_BACKEND",          "sqlite")
os.environ.setdefault("PHI_VAULT_SQLITE_PATH",  "/tmp/test_vault.sqlite")
os.environ.setdefault("AUDIT_BACKEND",          "sqlite")
os.environ.setdefault("AUDIT_SQLITE_PATH",      "/tmp/test_audit.sqlite")
os.environ.setdefault("LLM_PROVIDER",           "rules")
os.environ.setdefault("RAG_BACKEND",            "local")
os.environ.setdefault("AUTH_ENABLED",           "false")

print("Testing imports...")
try:
    from agent.security.token_vault import reset_vault; reset_vault()
    from agent.logging.audit_logger import reset_audit_logger; reset_audit_logger()
    from agent.api import app
    print("✓ All imports OK")
    print("✓ App created successfully")
    print("  Routes:", [r.path for r in app.routes])
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
