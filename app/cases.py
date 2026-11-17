"""Support cases: human escalation with a defined lifecycle. Every change is checked, recorded and audited."""
from . import db

TRANSITIONS = {
    "open": {"assigned", "cancelled"},
    "assigned": {"in_progress", "open", "cancelled"},
    "in_progress": {"awaiting_user", "resolved"},
    "awaiting_user": {"in_progress", "resolved", "cancelled"},
    "resolved": set(),
    "cancelled": set(),
}


class CaseError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def create_case(conn, user, question, summary, reason, idem_key):
    """Create a case once per idempotency key: a repeated click returns the same case."""
    if not idem_key or len(idem_key) > 64:
        raise CaseError("invalid_key")
    existing = conn.execute("SELECT * FROM cases WHERE idem_key=?", (idem_key,)).fetchone()
    if existing:
        if existing["user_id"] != user["id"]:
            raise CaseError("invalid_key")
        return dict(existing), False
    question, summary = (question or "").strip()[:1000], (summary or "").strip()[:2000]
    if not question or not summary:
        raise CaseError("invalid_input")
    cur = conn.execute("INSERT INTO cases (user_id, question, summary, reason, idem_key) VALUES (?,?,?,?,?)",
                       (user["id"], question, summary, reason or "user_requested", idem_key))
    number = f"CASE-2026-{cur.lastrowid:06d}"
    conn.execute("UPDATE cases SET case_number=? WHERE id=?", (number, cur.lastrowid))
    conn.execute("INSERT INTO case_events (case_id, actor_id, from_status, to_status) VALUES (?,?,?,?)",
                 (cur.lastrowid, user["id"], None, "open"))
    conn.commit()
    db.audit(conn, user["id"], "case_create", "ok", number)
    return dict(conn.execute("SELECT * FROM cases WHERE id=?", (cur.lastrowid,)).fetchone()), True


def list_cases(conn, user, status=None, unassigned=False):
    sql, args = "SELECT c.*, u.name AS requester FROM cases c JOIN users u ON u.id=c.user_id WHERE 1=1", []
    if user["role"] in ("member", "provider"):
        sql += " AND c.user_id=?"
        args.append(user["id"])
    if status:
        sql += " AND c.status=?"
        args.append(status)
    if unassigned:
        sql += " AND c.assigned_to IS NULL AND c.status='open'"
    return [dict(r) for r in conn.execute(sql + " ORDER BY c.id DESC", args)]


def change_status(conn, actor, case_id, new_status=None, assign_to=None):
    case = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not case:
        raise CaseError("not_found")
    if assign_to is not None:                                  # administrator assigns a case to an agent
        agent = conn.execute("SELECT id FROM users WHERE id=? AND role='agent'", (assign_to,)).fetchone()
        if not agent or case["status"] not in ("open", "assigned"):
            raise CaseError("invalid_transition")
        new_status, owner = "assigned", agent["id"]
    else:
        owner = case["assigned_to"]
        if new_status == "assigned":
            owner = actor["id"]                                 # an agent takes the case for themselves
    if new_status not in TRANSITIONS[case["status"]]:
        raise CaseError("invalid_transition")
    if new_status == "open":
        owner = None
    conn.execute("UPDATE cases SET status=?, assigned_to=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                 (new_status, owner, case_id))
    conn.execute("INSERT INTO case_events (case_id, actor_id, from_status, to_status) VALUES (?,?,?,?)",
                 (case_id, actor["id"], case["status"], new_status))
    conn.commit()
    db.audit(conn, actor["id"], "case_update", "ok", f"{case['case_number']}:{case['status']}->{new_status}")
    return dict(conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone())


def metrics(conn):
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]
    by_status = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) n FROM cases GROUP BY status")}
    outcomes = {r["outcome"]: r["n"] for r in conn.execute("SELECT outcome, COUNT(*) n FROM messages GROUP BY outcome")}
    gaps = [dict(r) for r in conn.execute(
        "SELECT question, COUNT(*) n FROM messages WHERE outcome='no_evidence' GROUP BY question ORDER BY n DESC LIMIT 10")]
    return {
        "questions_asked": one("SELECT COUNT(*) FROM messages"),
        "answer_outcomes": outcomes,
        "tool_calls": one("SELECT COUNT(*) FROM tool_invocations"),
        "access_denied": one("SELECT COUNT(*) FROM audit_log WHERE outcome='denied'"),
        "cases_by_status": by_status,
        "open_workload": sum(by_status.get(s, 0) for s in ("open", "assigned", "in_progress", "awaiting_user")),
        "knowledge_gaps": gaps,
    }
