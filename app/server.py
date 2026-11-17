"""Standard-library HTTP server (no web framework needed).

    python -m app.server            # http://127.0.0.1:8000

Every protected route checks the signed token, then the permission for the action.
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import auth, cases, config, db
from . import llm
from .assistant import Assistant
from .auth import can

STATIC = Path(__file__).parent / "static"
_failures = {}                      # email -> [timestamps of failed logins]
_lock = threading.Lock()


def rate_limited(email):
    now = time.time()
    with _lock:
        recent = [t for t in _failures.get(email, []) if now - t < config.LOGIN_WINDOW_SECONDS]
        _failures[email] = recent
        return len(recent) >= config.MAX_LOGIN_FAILURES


def note_failure(email):
    with _lock:
        _failures.setdefault(email, []).append(time.time())


class App:
    def __init__(self, conn=None, assistant=None):
        self.conn = conn or db.open_db()
        self.assistant = assistant or Assistant(llm=llm.from_env())
        self.lock = threading.Lock()         # one writer at a time keeps SQLite simple

    # Each method returns (status_code, body_dict)
    def login(self, body):
        email = str(body.get("email", "")).strip().lower()[:200]
        password = str(body.get("password", ""))[:200]
        if rate_limited(email):
            return 429, {"error": "Too many attempts. Try again later."}
        row = self.conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not row or not auth.verify_password(password, row["password_hash"]):
            note_failure(email)
            db.audit(self.conn, row["id"] if row else None, "login", "denied")
            return 401, {"error": "Email or password is not correct."}   # same message whether or not the email exists
        db.audit(self.conn, row["id"], "login", "ok")
        return 200, {"token": auth.make_token(row["id"]), "user": {"name": row["name"], "role": row["role"]}}

    def logout(self, user):
        self.conn.execute("INSERT OR IGNORE INTO revoked_tokens (jti) VALUES (?)", (user["jti"],))
        self.conn.commit()
        return 200, {"ok": True}

    def me(self, user):
        return 200, {"name": user["name"], "role": user["role"], "plan": user["plan"],
                     "member_number": user["member_number"], "permissions": sorted(auth.PERMISSIONS[user["role"]])}

    def chat(self, user, body):
        if not can(user, "ask"):
            return 403, {"error": "Not allowed"}
        message = body.get("message")
        if not isinstance(message, str) or not message.strip():
            return 400, {"error": "Please type a question."}
        if len(message) > config.MAX_MESSAGE_CHARS:
            return 400, {"error": f"Please keep questions under {config.MAX_MESSAGE_CHARS} characters."}
        out = self.assistant.respond(self.conn, user, message)
        out.pop("fell_back", None)
        return 200, out

    def claims(self, user):
        if not can(user, "read_own_claims"):
            db.audit(self.conn, user["id"], "claims_list", "denied")
            return 403, {"error": "Not allowed"}
        rows = self.conn.execute("SELECT * FROM claims WHERE member_number=? ORDER BY service_date DESC",
                                 (user["member_number"],)).fetchall()
        return 200, {"claims": [dict(r) for r in rows]}

    def create_case(self, user, body):
        if not can(user, "create_case"):
            db.audit(self.conn, user["id"], "case_create", "denied")
            return 403, {"error": "Not allowed"}
        try:
            case, created = cases.create_case(self.conn, user, body.get("question"), body.get("summary"),
                                              body.get("reason"), body.get("idempotency_key"))
        except cases.CaseError as e:
            return 400, {"error": e.code}
        return (201 if created else 200), {"case": case, "created": created}

    def list_cases(self, user, query):
        if not (can(user, "read_cases") or can(user, "read_own_cases")):
            return 403, {"error": "Not allowed"}
        status = query.get("status", [None])[0]
        payload = {"cases": cases.list_cases(self.conn, user, status, query.get("unassigned", ["0"])[0] == "1")}
        if can(user, "assign_case"):
            payload["agents"] = [dict(r) for r in self.conn.execute("SELECT id, name FROM users WHERE role='agent'")]
        return 200, payload

    def update_case(self, user, case_id, body):
        wants_assign = body.get("assign_to") is not None
        if not (can(user, "assign_case") if wants_assign else can(user, "update_case")):
            db.audit(self.conn, user["id"], "case_update", "denied")
            return 403, {"error": "Not allowed"}
        try:
            return 200, {"case": cases.change_status(self.conn, user, case_id, body.get("status"), body.get("assign_to"))}
        except cases.CaseError as e:
            return (404 if e.code == "not_found" else 400), {"error": e.code}

    def metrics(self, user):
        if not can(user, "view_metrics"):
            db.audit(self.conn, user["id"], "metrics", "denied")
            return 403, {"error": "Not allowed"}
        return 200, cases.metrics(self.conn)

    def audit_log(self, user):
        if not can(user, "read_audit"):
            db.audit(self.conn, user["id"], "audit_read", "denied")
            return 403, {"error": "Not allowed"}
        rows = self.conn.execute("SELECT ts, user_id, action, outcome, detail FROM audit_log ORDER BY id DESC LIMIT 100")
        return 200, {"entries": [dict(r) for r in rows]}

    # ---------------------------------------------------------------- routing
    def handle(self, method, path, query, body, token):
        with self.lock:
            if (method, path) == ("POST", "/api/login"):
                return self.login(body)
            user = auth.user_from_token(self.conn, token)
            if not user:
                return 401, {"error": "Please sign in."}
            if (method, path) == ("POST", "/api/logout"):
                return self.logout(user)
            if (method, path) == ("GET", "/api/me"):
                return self.me(user)
            if (method, path) == ("POST", "/api/chat"):
                return self.chat(user, body)
            if (method, path) == ("GET", "/api/claims"):
                return self.claims(user)
            if (method, path) == ("POST", "/api/cases"):
                return self.create_case(user, body)
            if (method, path) == ("GET", "/api/cases"):
                return self.list_cases(user, query)
            m = re.fullmatch(r"/api/cases/(\d+)", path)
            if method == "PATCH" and m:
                return self.update_case(user, int(m.group(1)), body)
            if (method, path) == ("GET", "/api/admin/metrics"):
                return self.metrics(user)
            if (method, path) == ("GET", "/api/admin/audit"):
                return self.audit_log(user)
            return 404, {"error": "Not found"}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _api(self, method):
            url = urlparse(self.path)
            body = {}
            if method in ("POST", "PATCH"):
                length = int(self.headers.get("Content-Length") or 0)
                if length > 20_000:
                    return self._send(413, {"error": "Request too large"})
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._send(400, {"error": "Invalid JSON"})
            token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
            try:
                status, payload = app.handle(method, url.path, parse_qs(url.query), body, token)
            except Exception:          # never leak internals to the browser
                status, payload = 500, {"error": "Something went wrong on our side."}
            self._send(status, payload)

        def do_GET(self):
            if self.path.startswith("/api/"):
                return self._api("GET")
            page = STATIC / "index.html"
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            self._api("POST")

        def do_PATCH(self):
            self._api("PATCH")

        def log_message(self, *args):
            pass
    return Handler


def main(host="127.0.0.1", port=8000):
    app = App()
    server = ThreadingHTTPServer((host, port), make_handler(app))
    print(f"Healthcare Support Agent (synthetic data) running at http://{host}:{port}")
    print("Language model:", "Gemini rewrite ON (answers are verified against the sources)" if llm.from_env() else "off (quoted answers only)")
    print("Demo logins: priya@example.test, ravi@example.test, jordan@example.test, sam@example.test, alex@example.test  /  password: demo123")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
