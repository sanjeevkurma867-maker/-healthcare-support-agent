"""Allow-listed tools. Every call is schema-checked, permission-checked, budgeted and audited."""
import json
import re
from . import config, db
from .auth import can

SPECS = {
    "list_my_claims": {"permission": "read_own_claims", "args": {}},
    "claim_status": {"permission": "read_own_claims", "args": {"claim_number": r"^SYN-CLM-\d{4}-\d{6}$"}},
    "plan_benefits": {"permission": "read_plan", "args": {"plan_code": r"^[A-Z]{2,12}$"}, "optional": {"plan_code"}},
    "required_documents": {"permission": "lookup_requirements", "args": {"procedure": r"^.{1,200}$"}},
}


class ToolExecutor:
    def __init__(self, conn, user):
        self.conn, self.user, self.calls = conn, user, 0

    def _record(self, name, outcome):
        self.conn.execute("INSERT INTO tool_invocations (user_id, tool, outcome) VALUES (?,?,?)",
                          (self.user["id"], name, outcome))
        db.audit(self.conn, self.user["id"], f"tool:{name}", outcome)

    def run(self, name, args=None):
        args = args or {}
        spec = SPECS.get(name)
        if spec is None:
            self._record(str(name)[:40], "denied")
            return {"ok": False, "error": "not_allowed"}
        self.calls += 1
        if self.calls > config.TOOL_CALL_BUDGET:
            self._record(name, "denied")
            return {"ok": False, "error": "budget_exceeded"}
        unknown = set(args) - set(spec["args"])
        missing = {k for k in spec["args"] if k not in args and k not in spec.get("optional", set())}
        if unknown or missing or any(not isinstance(v, str) or not re.match(spec["args"][k], v) for k, v in args.items()):
            self._record(name, "invalid")
            return {"ok": False, "error": "invalid_args"}
        if not can(self.user, spec["permission"]):
            self._record(name, "denied")
            return {"ok": False, "error": "forbidden"}
        result = getattr(self, f"_{name}")(**args)
        self._record(name, "ok" if result["ok"] else result["error"])
        return result

    # -- tools --------------------------------------------------------------
    def _list_my_claims(self):
        rows = self.conn.execute("SELECT * FROM claims WHERE member_number=? ORDER BY service_date DESC",
                                 (self.user["member_number"],)).fetchall()
        return {"ok": True, "data": [dict(r) for r in rows]}

    def _claim_status(self, claim_number):
        # Ownership is part of the query: another member's claim looks exactly like a claim that does not exist.
        row = self.conn.execute("SELECT * FROM claims WHERE claim_number=? AND member_number=?",
                                (claim_number, self.user["member_number"])).fetchone()
        return {"ok": True, "data": dict(row)} if row else {"ok": False, "error": "not_found"}

    def _plan_benefits(self, plan_code=None):
        code = self.user["plan"] if self.user["role"] == "member" else plan_code
        if self.user["role"] == "member" and plan_code and plan_code != self.user["plan"]:
            return {"ok": False, "error": "not_found"}
        if not code:
            return {"ok": False, "error": "not_found"}
        row = self.conn.execute("SELECT * FROM plans WHERE code=?", (code,)).fetchone()
        return {"ok": True, "data": dict(row)} if row else {"ok": False, "error": "not_found"}

    def _required_documents(self, procedure):
        low = procedure.lower()
        for s in self.conn.execute("SELECT * FROM requirement_sets").fetchall():
            if any(k in low for k in json.loads(s["keywords"])):
                reqs = [r["text"] for r in self.conn.execute(
                    "SELECT text FROM requirements WHERE set_code=? ORDER BY position", (s["code"],))]
                return {"ok": True, "data": {"code": s["code"], "name": s["name"], "requirements": reqs}}
        return {"ok": False, "error": "not_found"}
