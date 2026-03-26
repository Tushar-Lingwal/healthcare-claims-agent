"""
encryption.py — AES-256-GCM encryption for PHI vault values.

AES-256-GCM provides confidentiality (256-bit key), integrity (auth tag),
and unique IV per encryption. Key loaded from PHI_VAULT_KEY env var only.

Generate a key: python -c "import secrets; print(secrets.token_hex(32))"
"""

import os
import base64
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_ENV_VAR   = "PHI_VAULT_KEY"
_IV_SIZE_BYTES = 12   # 96-bit IV — standard for AES-GCM


class EncryptionKeyError(Exception):
    """Raised when the encryption key is missing or invalid."""

class DecryptionError(Exception):
    """Raised when decryption fails — tampered data or wrong key."""


def _load_key() -> bytes:
    raw = os.environ.get(_KEY_ENV_VAR, "").strip()
    if not raw:
        raise EncryptionKeyError(
            f"PHI vault key not found. Set {_KEY_ENV_VAR} in .env.\n"
            f"Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    try:
        key_bytes = bytes.fromhex(raw)
    except ValueError:
        raise EncryptionKeyError(f"{_KEY_ENV_VAR} must be a 64-char hex string.")
    if len(key_bytes) != 32:
        raise EncryptionKeyError(
            f"{_KEY_ENV_VAR} must be 32 bytes (64 hex chars). Got {len(key_bytes)}."
        )
    return key_bytes


def encrypt(plaintext: str) -> str:
    """
    Encrypts plaintext using AES-256-GCM.
    Returns base64url string: <12-byte IV><ciphertext+auth tag>
    Each call produces a unique ciphertext even for identical inputs.
    """
    key    = _load_key()
    iv     = secrets.token_bytes(_IV_SIZE_BYTES)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    return base64.urlsafe_b64encode(iv + ciphertext).decode("ascii")


def decrypt(encrypted_blob: str) -> str:
    """
    Decrypts a base64url AES-256-GCM blob back to plaintext.
    Raises DecryptionError if tampered, corrupted, or wrong key.
    """
    key = _load_key()
    try:
        blob = base64.urlsafe_b64decode(encrypted_blob.encode("ascii"))
    except Exception:
        raise DecryptionError("Invalid base64 encoding in encrypted blob.")
    if len(blob) < _IV_SIZE_BYTES + 16:
        raise DecryptionError(f"Blob too short ({len(blob)} bytes).")
    iv         = blob[:_IV_SIZE_BYTES]
    ciphertext = blob[_IV_SIZE_BYTES:]
    try:
        return AESGCM(key).decrypt(iv, ciphertext, None).decode("utf-8")
    except Exception:
        raise DecryptionError("Decryption failed — data tampered or wrong key.")


def encrypt_if_present(value) -> str | None:
    """Encrypts value if not None."""
    return encrypt(value) if value is not None else None


def decrypt_if_present(blob) -> str | None:
    """Decrypts blob if not None."""
    return decrypt(blob) if blob is not None else None


def is_key_configured() -> bool:
    """Returns True if PHI_VAULT_KEY is set and valid. Use in health checks."""
    try:
        _load_key()
        return True
    except EncryptionKeyError:
        return False


def generate_key() -> str:
    """Generates a new random 32-byte AES-256 key as hex string."""
    return secrets.token_hex(32)
