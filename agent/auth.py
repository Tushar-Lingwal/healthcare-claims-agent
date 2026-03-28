"""
auth.py — JWT authentication for the Claims Adjudication API.

Simple but production-ready:
  - Users stored in database (SQLite local, PostgreSQL production)
  - Passwords hashed with bcrypt
  - JWT tokens with configurable expiry
  - Three roles: admin, reviewer, submitter

Endpoints added to api.py:
  POST /auth/login    — returns JWT token
  POST /auth/register — creates new user (admin only in production)
  GET  /auth/me       — returns current user info

Usage in protected routes:
  @app.post("/adjudicate")
  async def adjudicate(req, current_user=Depends(get_current_user)):
      ...
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# JWT — graceful import
try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SECRET_KEY    = os.environ.get("SECRET_KEY", "dev-secret-change-in-production-please")
ALGORITHM     = "HS256"
TOKEN_EXPIRY  = int(os.environ.get("TOKEN_EXPIRY_HOURS", "24"))

_AUTH_DB_PATH = Path(os.environ.get("AUTH_DB_PATH", "./auth.sqlite"))
_bearer       = HTTPBearer(auto_error=False)

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role:     str = "submitter"   # admin | reviewer | submitter

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    username:     str
    role:         str
    expires_in:   int

class UserInfo(BaseModel):
    username: str
    role:     str
    created_at: str

# ─────────────────────────────────────────────
# USER DATABASE
# ─────────────────────────────────────────────

_db_lock = threading.Lock()

def _get_auth_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_AUTH_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role       TEXT NOT NULL DEFAULT 'submitter',
            created_at TEXT NOT NULL,
            is_active  INTEGER DEFAULT 1
        );
    """)
    conn.commit()
    return conn


def _hash_password(password: str) -> str:
    if not _AUTH_AVAILABLE:
        return f"plain:{password}"  # fallback for dev without passlib
    password = password[:72]  # bcrypt 72-byte hard limit
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return ctx.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    if not _AUTH_AVAILABLE:
        return hashed == f"plain:{plain}"
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return ctx.verify(plain, hashed)


def create_user(username: str, password: str, role: str = "submitter") -> bool:
    """Creates a new user. Returns False if username already exists."""
    conn = _get_auth_db()
    try:
        with _db_lock:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username, _hash_password(password), role, datetime.utcnow().isoformat())
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username: str) -> Optional[dict]:
    conn = _get_auth_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def ensure_default_admin():
    """Creates a default admin user if no users exist."""
    try:
        conn = _get_auth_db()
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        if count == 0:
            default_pass = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")[:72]
            create_user("admin", default_pass, "admin")
            print("\n  [auth] Default admin created: username=admin")
    except Exception as e:
        print(f"  [auth] Skipped admin creation: {e}")

# ─────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────

def create_token(username: str, role: str) -> str:
    if not _AUTH_AVAILABLE:
        # Simple base64 token for dev without python-jose
        import base64, json
        payload = {"sub": username, "role": role, "exp": "dev"}
        return base64.b64encode(json.dumps(payload).encode()).decode()

    payload = {
        "sub":  username,
        "role": role,
        "exp":  datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY),
        "iat":  datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    if not _AUTH_AVAILABLE:
        import base64, json
        try:
            return json.loads(base64.b64decode(token.encode()).decode())
        except:
            return None
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

# ─────────────────────────────────────────────
# FastAPI DEPENDENCIES
# ─────────────────────────────────────────────

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Optional[dict]:
    """
    Dependency — validates JWT and returns current user.
    If AUTH_ENABLED=false (default for local dev), always passes.
    Set AUTH_ENABLED=true in production.
    """
    if not AUTH_ENABLED:
        return {"username": "dev", "role": "admin"}

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user(payload.get("sub", ""))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    return user


def require_role(*roles: str):
    """Dependency factory — requires one of the given roles."""
    async def check(user=Depends(get_current_user)):
        if not AUTH_ENABLED:
            return user
        if user["role"] not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' not authorized. Required: {roles}",
            )
        return user
    return check


# ─────────────────────────────────────────────
# AUTH ROUTER (mounted in api.py)
# ─────────────────────────────────────────────

from fastapi import APIRouter

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])


@auth_router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """
    Login with username and password.
    Returns a JWT token valid for TOKEN_EXPIRY_HOURS (default 24h).
    """
    if not AUTH_ENABLED:
        # Dev mode — any login works
        return TokenResponse(
            access_token=create_token(req.username, "admin"),
            username=req.username,
            role="admin",
            expires_in=TOKEN_EXPIRY * 3600,
        )

    user = get_user(req.username)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_token(req.username, user["role"])
    return TokenResponse(
        access_token=token,
        username=req.username,
        role=user["role"],
        expires_in=TOKEN_EXPIRY * 3600,
    )


@auth_router.post("/register")
async def register(
    req: RegisterRequest,
    current_user=Depends(require_role("admin")),
):
    """Creates a new user. Admin only."""
    if req.role not in ("admin", "reviewer", "submitter"):
        raise HTTPException(status_code=400, detail="Invalid role")

    success = create_user(req.username, req.password, req.role)
    if not success:
        raise HTTPException(status_code=409, detail="Username already exists")

    return {"message": f"User '{req.username}' created with role '{req.role}'"}


@auth_router.get("/me", response_model=UserInfo)
async def me(current_user=Depends(get_current_user)):
    """Returns current authenticated user info."""
    return UserInfo(
        username   = current_user["username"],
        role       = current_user["role"],
        created_at = current_user.get("created_at", ""),
    )


@auth_router.get("/status")
async def auth_status():
    """Returns whether authentication is enabled."""
    return {
        "auth_enabled": AUTH_ENABLED,
        "message": "Auth disabled (dev mode)" if not AUTH_ENABLED else "Auth enabled"
    }