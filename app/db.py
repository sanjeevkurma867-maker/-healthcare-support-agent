"""SQLite schema and seeding. Constraints live in the database as well as in the code."""
import hashlib
import json
import os
import sqlite3
from . import config

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('member','provider','agent','admin')),
  password_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plans (
  code TEXT PRIMARY KEY, name TEXT NOT NULL, deductible INTEGER, oop_max INTEGER,
  pcp_copay INTEGER, specialist_copay INTEGER, urgent_copay INTEGER, er_copay INTEGER,
  generic_rx_copay INTEGER, brand_rx_copay INTEGER, coinsurance_pct INTEGER, virtual_copay INTEGER);
CREATE TABLE IF NOT EXISTS members (
  member_number TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
  plan_code TEXT NOT NULL REFERENCES plans(code));
CREATE TABLE IF NOT EXISTS claims (
  claim_number TEXT PRIMARY KEY, member_number TEXT NOT NULL REFERENCES members(member_number),
  provider TEXT, service_date TEXT, billed REAL, plan_paid REAL, member_owes REAL,
  status TEXT NOT NULL CHECK (status IN ('Received','In Review','Pending Information','Processed - Paid','Processed - Denied')),
  reason TEXT);
CREATE TABLE IF NOT EXISTS requirement_sets (code TEXT PRIMARY KEY, name TEXT NOT NULL, keywords TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requirements (
  id INTEGER PRIMARY KEY, set_code TEXT NOT NULL REFERENCES requirement_sets(code), position INTEGER, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revoked_tokens (jti TEXT PRIMARY KEY, revoked_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), question TEXT NOT NULL,
  route TEXT, outcome TEXT, sources TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS tool_invocations (
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, tool TEXT NOT NULL, outcome TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS cases (
  id INTEGER PRIMARY KEY, case_number TEXT UNIQUE, user_id INTEGER NOT NULL REFERENCES users(id),
  question TEXT NOT NULL, summary TEXT NOT NULL, reason TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','assigned','in_progress','awaiting_user','resolved','cancelled')),
  assigned_to INTEGER REFERENCES users(id), idem_key TEXT UNIQUE,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS case_events (
  id INTEGER PRIMARY KEY, case_id INTEGER NOT NULL REFERENCES cases(id), actor_id INTEGER,
  from_status TEXT, to_status TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, user_id INTEGER, action TEXT NOT NULL,
  outcome TEXT NOT NULL, detail TEXT);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
  BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""

DEMO_PASSWORD = "demo123"   # synthetic accounts only


def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def connect(path=None):
    conn = sqlite3.connect(str(path or config.DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def seed(conn):
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
        return
    with open(config.DATA_DIR / "seed.json", encoding="utf-8") as f:
        data = json.load(f)
    for p in data["plans"]:
        conn.execute("INSERT INTO plans VALUES (:code,:name,:deductible,:oop_max,:pcp_copay,:specialist_copay,"
                     ":urgent_copay,:er_copay,:generic_rx_copay,:brand_rx_copay,:coinsurance_pct,:virtual_copay)", p)
    pw = hash_password(DEMO_PASSWORD)
    for u in data["users"]:
        cur = conn.execute("INSERT INTO users (email,name,role,password_hash) VALUES (?,?,?,?)",
                           (u["email"], u["name"], u["role"], pw))
        if u["role"] == "member":
            conn.execute("INSERT INTO members VALUES (?,?,?)", (u["member_number"], cur.lastrowid, u["plan"]))
    for c in data["claims"]:
        conn.execute("INSERT INTO claims VALUES (:claim_number,:member_number,:provider,:service_date,:billed,"
                     ":plan_paid,:member_owes,:status,:reason)", c)
    for s in data["requirement_sets"]:
        conn.execute("INSERT INTO requirement_sets VALUES (?,?,?)", (s["code"], s["name"], json.dumps(s["keywords"])))
        for i, text in enumerate(s["requirements"], 1):
            conn.execute("INSERT INTO requirements (set_code,position,text) VALUES (?,?,?)", (s["code"], i, text))
    conn.commit()


def open_db(path=None):
    conn = connect(path)
    init_db(conn)
    seed(conn)
    return conn


def audit(conn, user_id, action, outcome, detail=""):
    """Append one audit record. Details hold identifiers and counts, never message text."""
    conn.execute("INSERT INTO audit_log (user_id, action, outcome, detail) VALUES (?,?,?,?)",
                 (user_id, action, outcome, detail))
    conn.commit()
