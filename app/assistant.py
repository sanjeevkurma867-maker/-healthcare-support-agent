"""The assistant pipeline.

question -> router -> (refuse | tool | human handoff | permission-filtered retrieval)
retrieval -> evidence check -> quoted answer with citations -> citation validation
No evidence -> "I won't guess" + an offer to open a support case.
"""
import json
import re
from . import config, db
from . import llm as llm_module
from .retrieval import Retriever
from .router import route
from .tools import ToolExecutor

REFUSALS = {
    "refuse_medical": ("I can't give medical advice, diagnoses or medicine guidance. Please talk to your doctor or "
                       "pharmacist. If this is an emergency, call your local emergency number. I can help with "
                       "benefits, claims and paperwork, or pass your question to a support representative."),
    "refuse_decision": ("I can't approve, deny or change a claim or an authorization. Those decisions are made by the "
                        "plan's reviewers. I can explain how the process works, or ask a support representative to "
                        "look at your case."),
    "refuse_privacy": ("I can only show information from your own records. I can't share another person's claims, "
                       "plan or personal details."),
    "injection": ("I can't follow that request. I only answer questions about benefits, claims and prior "
                  "authorization, and I follow the same rules for everyone."),
}
NO_EVIDENCE = ("I couldn't find this in the plan documents available to you, so I won't guess. "
               "I can ask a support representative to follow up.")
GREETING_TEXT = ("Hello! I can explain plan benefits, look up your own claims, and describe what prior authorization "
                 "paperwork needs. Try asking: \"What is my deductible?\", \"Show my claims\" or \"How do I appeal a denied claim?\". "
                 "I don't give medical advice, and I never approve or deny anything. I can also pass your question to a person.")
FOOTER = "I give administrative information only, not medical advice. You can ask for simpler wording or for a person at any time."

LABELS = {"deductible": "Deductible", "oop_max": "Out-of-pocket maximum", "pcp_copay": "Primary care visit copay",
          "specialist_copay": "Specialist visit copay", "urgent_copay": "Urgent care copay", "er_copay": "Emergency room copay",
          "generic_rx_copay": "Generic drug copay (30 days)", "brand_rx_copay": "Preferred brand drug copay (30 days)",
          "virtual_copay": "Virtual visit copay", "coinsurance_pct": "Coinsurance after the deductible"}
GLOSSARY_ALIASES = {"eob": "explanation of benefits", "ppo": "", "oop": "out-of-pocket maximum"}


def money(x):
    return f"${x:,.2f}" if isinstance(x, float) else f"${x:,}"


def has_evidence(info, best_cosine):
    """Answer only when the evidence is strong enough. Two ways to qualify:
    A) specific: every question term is found in the top chunk and one sentence matches closely
       (short definition questions such as "What is a copay?");
    B) broad: enough question terms with enough weight are found, and one sentence matches well enough."""
    if not info["hits"]:
        return False
    specific = info["coverage"] >= config.SPECIFIC_COVERAGE and best_cosine >= config.SPECIFIC_COSINE
    broad = (best_cosine >= config.MIN_COSINE and info["matched_mass"] >= config.MIN_MASS
             and info["n_matched"] >= config.MIN_MATCHED)
    return specific or broad


def validate_citations(text, sources, readable_ids):
    """Every [Cn] marker must point at a retrieved source the user is allowed to read."""
    refs = {s["ref"] for s in sources}
    markers = set(re.findall(r"\[(C\d+)\]", text))
    if not markers <= refs:
        return False
    return all(s.get("chunk_id") in readable_ids for s in sources if s["ref"].startswith("C"))


class Assistant:
    def __init__(self, retriever=None, llm=None):
        self.retriever = retriever or Retriever()
        self.llm = llm                      # optional GeminiClient; None means quoted answers only
        self.llm_log = []                   # one status per attempted rewrite: ok / rejected reason / failed

    # ------------------------------------------------------------------ main
    def respond(self, conn, user, message):
        message = (message or "").strip()[: config.MAX_MESSAGE_CHARS]
        r = route(message, user)
        tools = ToolExecutor(conn, user)
        out = self._dispatch(conn, user, message, r, tools)
        out.setdefault("sources", [])
        out.setdefault("proposal", None)
        out["route"] = "retrieval" if out.get("fell_back") else (r["kind"] if r["kind"] != "tool" else f"tool:{r['tool']}")
        self._log(conn, user, message, out)
        return out

    def _dispatch(self, conn, user, message, r, tools):
        kind = r["kind"]
        if kind == "empty":
            return {"outcome": "refused", "answer": "Please type a question about your benefits, claims or paperwork."}
        if kind == "greeting":
            return {"outcome": "info", "answer": GREETING_TEXT}
        if kind in REFUSALS:
            out = {"outcome": "refused", "answer": REFUSALS[kind]}
            if kind in ("refuse_decision", "refuse_medical"):
                out["proposal"] = self._proposal(conn, user, message, "decision_or_medical_request")
            return out
        if kind == "human":
            return {"outcome": "human_requested",
                    "answer": "Of course. I've prepared a short summary so you don't have to repeat yourself. "
                              "Review it and confirm to send it to a support representative.",
                    "proposal": self._proposal(conn, user, message, "user_requested_person")}
        if kind == "tool":
            res = tools.run(r["tool"], r["args"])
            if res["ok"]:
                return self._format_tool(conn, user, r, res["data"])
            if res["error"] == "not_found" and r.get("fallback") == "retrieval":
                out = self._retrieval(conn, user, message)
                out["fell_back"] = True
                return out
            return self._tool_error(conn, user, message, r["tool"], res["error"])
        return self._retrieval(conn, user, message)

    # ------------------------------------------------------------- retrieval
    def _retrieval(self, conn, user, message):
        info = self.retriever.search(message, user)
        sents = self.retriever.best_sentences(info)
        best = max((s["cosine"] for s in sents), default=0.0)
        self.last_trace = {"coverage": info["coverage"], "matched_mass": info["matched_mass"],
                           "n_matched": info["n_matched"], "best_cosine": best,
                           "retrieved": [h["chunk"]["doc_id"] for h in info["hits"]]}
        if not has_evidence(info, best) or not sents:
            return {"outcome": "no_evidence", "answer": NO_EVIDENCE,
                    "proposal": self._proposal(conn, user, message, "no_adequate_answer")}
        sources, lines, ref_of = [], [], {}
        for s in sents:
            c = s["chunk"]
            if c["id"] not in ref_of:
                ref_of[c["id"]] = f"C{len(ref_of) + 1}"
                sources.append({"ref": ref_of[c["id"]], "chunk_id": c["id"], "doc_id": c["doc_id"], "title": c["title"],
                                "section": c["section"], "version": c["version"], "passage": c["text"]})
            prefix = f"{c['plan'].title()} plan: " if c["plan"] and user["role"] != "member" else ""
            lines.append(f"- {prefix}{s['sentence']} [{ref_of[c['id']]}]")
        readable_ids = {c["id"] for c in self.retriever.readable(user)}
        text = ("Here's what I found in your plan documents:" if user["role"] == "member"
                else "Here's what the documents say:") + "\n" + "\n".join(lines)
        if not validate_citations(text, sources, readable_ids):
            return {"outcome": "no_evidence", "answer": NO_EVIDENCE,
                    "proposal": self._proposal(conn, user, message, "citation_check_failed")}
        generated_by = "extractive"
        rewritten = self._rewrite(message, sources)
        if rewritten:
            text, generated_by = rewritten, "gemini"
        notes = self._term_notes(message, [s["sentence"] for s in sents], {s["doc_id"] for s in sources})
        if notes:
            text += "\n\nWhat these words mean:\n" + "\n".join(f"- {n}" for n in notes)
        return {"outcome": "answered", "answer": text + "\n\n" + FOOTER, "sources": sources, "generated_by": generated_by}

    def _rewrite(self, message, sources):
        """Ask the language model to phrase the grounded answer; accept it only if it passes verification."""
        if not self.llm:
            return None
        text = self.llm.generate(llm_module.SYSTEM_PROMPT, llm_module.build_prompt(message, sources))
        if text is None:
            self.llm_log.append("failed")
            return None
        ok, reason = llm_module.verify_answer(text, sources, message)
        self.llm_log.append("ok" if ok else f"rejected:{reason}")
        return text if ok else None

    def _term_notes(self, message, sentences, used_docs):
        if "glossary" in used_docs:
            return []
        text = (message + " " + " ".join(sentences)).lower()
        for alias, full in GLOSSARY_ALIASES.items():
            if full and re.search(rf"\b{alias}\b", text):
                text += " " + full
        notes = []
        for c in self.retriever.chunks:
            if c["doc_id"] == "glossary" and c["section"].lower() in text:
                first = re.split(r"(?<=[.!?])\s", c["text"])[0]
                notes.append(f"{c['section']}: {first}")
        return notes[:2]

    # ----------------------------------------------------------------- tools
    def _claims_meaning(self, status):
        for c in self.retriever.chunks:
            if c["doc_id"] == "claims-guide" and c["section"] == "Claim statuses":
                for sentence in re.split(r"(?<=[.!?])\s+", c["text"]):
                    if sentence.startswith(status):
                        return sentence, {"ref": "C1", "chunk_id": c["id"], "doc_id": c["doc_id"], "title": c["title"],
                                          "section": c["section"], "version": c["version"], "passage": c["text"]}
        return None, None

    def _format_tool(self, conn, user, r, data):
        tool = r["tool"]
        if tool == "claim_status":
            lines = [f"Claim {data['claim_number']} is {data['status']}.",
                     f"- Provider: {data['provider']}, service date {data['service_date']}",
                     f"- Billed {money(data['billed'])}; plan paid {money(data['plan_paid'])}; you owe {money(data['member_owes'])}"]
            if data["reason"]:
                lines.append(f"- Note on the claim: {data['reason']}")
            meaning, src = self._claims_meaning(data["status"])
            sources = [{"ref": "R1", "title": "Your claim record", "section": data["claim_number"],
                        "version": "live record", "passage": "\n".join(lines[1:])}]
            if meaning:
                lines.append(f"What this status means: {meaning} [C1]")
                sources.append(src)
            if data["status"] == "Processed - Denied":
                lines.append("If you disagree, you can appeal within 180 calendar days of the EOB date. Ask me how appeals work.")
            return {"outcome": "tool", "tool": tool, "answer": "\n".join(lines) + "\n\n" + FOOTER, "sources": sources}
        if tool == "list_my_claims":
            if not data:
                return {"outcome": "tool", "tool": tool, "answer": "You have no claims on record yet."}
            lines = ["Your claims, newest first:"] + [
                f"- {c['claim_number']}: {c['status']}, {c['provider']}, you owe {money(c['member_owes'])}" for c in data]
            return {"outcome": "tool", "tool": tool, "answer": "\n".join(lines) + "\n\nAsk about one claim number for more detail.",
                    "sources": [{"ref": "R1", "title": "Your claim records", "section": "", "version": "live record",
                                 "passage": "\n".join(lines[1:])}]}
        if tool == "plan_benefits":
            focus = r.get("focus")
            fields = [focus] if focus else list(LABELS)
            shown = [f"- {LABELS[f]}: " + (f"{data[f]}%" if f == "coinsurance_pct" else money(data[f])) for f in fields]
            head = (f"Your {data['name']} plan:" if user["role"] == "member" else f"{data['name']} plan:")
            return {"outcome": "tool", "tool": tool, "answer": head + "\n" + "\n".join(shown) + "\n\n" + FOOTER,
                    "sources": [{"ref": "R1", "title": f"Plan record: {data['name']}", "section": "", "version": "2026 record",
                                 "passage": "\n".join(shown)}]}
        items = [f"{i}. {t}" for i, t in enumerate(data["requirements"], 1)]
        return {"outcome": "tool", "tool": tool,
                "answer": f"Documentation checklist for {data['name']}:\n" + "\n".join(items) +
                          "\n\nThis checklist covers documentation only. It is not an approval or a denial.",
                "sources": [{"ref": "R1", "title": f"Requirement set: {data['name']}", "section": data["code"],
                             "version": "1.0", "passage": "\n".join(items)}]}

    def _tool_error(self, conn, user, message, tool, error):
        if error == "not_found":
            return {"outcome": "tool", "tool": tool,
                    "answer": "I couldn't find that in your records. Please check the number, or I can ask a support "
                              "representative to help.",
                    "proposal": self._proposal(conn, user, message, "record_not_found")}
        if error == "forbidden":
            return {"outcome": "refused", "tool": tool,
                    "answer": "Your account type can't use that lookup. I can answer general questions from the plan documents."}
        return {"outcome": "refused", "tool": tool, "answer": "I couldn't complete that lookup. Please rephrase your question."}

    # -------------------------------------------------------------- escalation
    def _proposal(self, conn, user, message, reason):
        earlier = [r["question"] for r in conn.execute(
            "SELECT question FROM messages WHERE user_id=? ORDER BY id DESC LIMIT 2", (user["id"],))][::-1]
        summary = f"Question: {message}."
        if earlier:
            summary += " Earlier questions: " + " | ".join(earlier) + "."
        summary += f" Reason for escalation: {reason.replace('_', ' ')}."
        return {"type": "support_case", "question": message, "summary": summary, "reason": reason}

    # ----------------------------------------------------------------- logging
    def _log(self, conn, user, message, out):
        doc_ids = sorted({s["doc_id"] for s in out["sources"] if "doc_id" in s})
        conn.execute("INSERT INTO messages (user_id, question, route, outcome, sources) VALUES (?,?,?,?,?)",
                     (user["id"], message, out["route"], out["outcome"], json.dumps(doc_ids)))
        conn.commit()
        denied = out["route"] in ("refuse_privacy", "injection") or (out["outcome"] == "refused" and out.get("tool"))
        db.audit(conn, user["id"], "ask", "denied" if denied else "ok", f"{out['route']}/{out['outcome']}")
