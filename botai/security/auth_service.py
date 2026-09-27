"""Authentication helpers: password hashing, session lookup va default admin."""

import hashlib
import hmac
import os
import secrets

from botai.services.database_service import _db_connect, _now_iso

__all__ = [
    "_hash_password",
    "_verify_password",
    "_ensure_default_admin_user",
    "_user_from_token",
    "_create_auth_session",
]

def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt, digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    candidate = _hash_password(password, salt).split("$", 2)[2]
    return hmac.compare_digest(candidate, digest)


def _ensure_default_admin_user() -> None:
    username = os.getenv("ADMIN_USERNAME", "").strip()
    password = os.getenv("ADMIN_PASSWORD", "").strip()
    if not username or not password:
        return
    with _db_connect() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            return
        conn.execute(
            """
            INSERT INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (username, _hash_password(password), "admin", _now_iso()),
        )
        conn.commit()


def _user_from_token(token: str | None) -> dict | None:
    if not token:
        return None
    with _db_connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.username, users.role, auth_sessions.created_at AS session_created_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token = ?
            """,
            (token.strip(),),
        ).fetchone()
    return _row_to_dict(row) if row else None


def _create_auth_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _db_connect() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _now_iso()),
        )
        conn.commit()
    return token

