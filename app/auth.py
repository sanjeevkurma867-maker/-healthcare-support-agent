"""Password checking, signed tokens, and the role -> permission map."""
import base64
import hashlib
import hmac
import json
import time
import uuid
from . import config

PERMISSIONS = {
    "member": {"ask", "read_own_claims", "read_plan", "create_case", "read_own_cases", "lookup_requirements"},
    "provider": {"ask", "read_plan", "create_case", "read_own_cases", "lookup_requirements"},
    "agent": {"ask", "read_plan", "read_cases", "update_case", "lookup_requirements"},
    "admin": {"ask", "read_plan", "read_cases", "assign_case", "view_metrics", "read_audit"},
}

# Which document audiences each role may retrieve from.
DOC_AUDIENCES = {
    "member": {"member"},
    "provider": {"member", "provider"},
    "agent": {"member", "provider", "internal"},
    "admin": {"member", "provider", "internal"},
}


def can(user, permission: str) -> bool:
    return permission in PERMISSIONS.get(user["role"], set())


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except ValueError:
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def make_token(user_id: int) -> str:
    payload = {"uid": user_id, "exp": int(time.time()) + config.TOKEN_TTL_SECONDS, "jti": uuid.uuid4().hex}
    body = _b64(json.dumps(payload).encode())
    sig = _b64(hmac.new(config.SECRET.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_token(token: str):
    """Return the payload dict, or None if the token is forged, malformed or expired."""
    try:
        body, sig = token.split(".")
        expected = _b64(hmac.new(config.SECRET.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload["exp"] < time.time():
            return None
        return payload
    except Exception:
        return None


def user_from_token(conn, token: str):
    payload = read_token(token or "")
    if not payload:
        return None
    if conn.execute("SELECT 1 FROM revoked_tokens WHERE jti=?", (payload["jti"],)).fetchone():
        return None
    row = conn.execute("SELECT id,email,name,role FROM users WHERE id=?", (payload["uid"],)).fetchone()
    if not row:
        return None
    user = dict(row)
    m = conn.execute("SELECT member_number, plan_code FROM members WHERE user_id=?", (user["id"],)).fetchone()
    user["member_number"] = m["member_number"] if m else None
    user["plan"] = m["plan_code"] if m else None
    user["jti"] = payload["jti"]
    return user
