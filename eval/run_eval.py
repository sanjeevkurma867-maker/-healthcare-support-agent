"""Evaluate the assistant on the synthetic question set.

    python eval/run_eval.py            # report results on both splits
    python eval/run_eval.py --tune     # grid-search thresholds on the DEV split only

Thresholds are tuned on the dev split; the test split is only reported, never tuned on.
"""
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import auth, config, db, llm       # noqa: E402
from app.assistant import Assistant         # noqa: E402

with open(Path(__file__).parent / "questions.json", encoding="utf-8") as _f:
    QUESTIONS = json.load(_f)


USE_LLM = "--llm" in sys.argv          # needs HSA_LLM=gemini and GEMINI_API_KEY in the environment
LLM_LOG = []


def setup():
    conn = db.open_db(os.path.join(tempfile.mkdtemp(), "eval.db"))
    users = {}
    for key, email in QUESTIONS["users"].items():
        uid = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()["id"]
        users[key] = auth.user_from_token(conn, auth.make_token(uid))
    return conn, users, Assistant(llm=llm.from_env() if USE_LLM else None)


def judge(item, out, trace):
    """Return (behaviour_ok, details)."""
    text = out["answer"].lower()
    cited = {s.get("doc_id") for s in out["sources"] if s.get("doc_id")}
    forbidden_hit = bool(cited & set(item["forbid_docs"])) or any(f in text for f in item["forbid_docs"] if f.startswith("in "))
    facts_ok = all(f in text for f in item["facts"])
    t = item["type"]
    if t == "answer":
        ok = out["outcome"] == "answered" and bool(cited & set(item["docs"])) and facts_ok and not forbidden_hit
    elif t == "tool":
        ok = out["route"] == f"tool:{item['tool']}" and out["outcome"] == "tool" and facts_ok
    elif t == "access":
        ok = out["route"] == f"tool:{item['tool']}" and facts_ok and not forbidden_hit
    elif t == "refuse":
        ok = out["outcome"] == "refused"
    elif t == "no_answer":
        ok = out["outcome"] == "no_evidence" and not forbidden_hit
    else:  # human
        ok = out["outcome"] == "human_requested"
    return ok, {"facts_ok": facts_ok, "cited": sorted(cited), "forbidden_hit": forbidden_hit}


def run(split=None, quiet=True):
    conn, users, assistant = setup()
    rows, latencies = [], []
    for item in QUESTIONS["items"]:
        if split and item["split"] != split:
            continue
        assistant.last_trace = {}
        start = time.perf_counter()
        calls_before = len(assistant.llm_log)
        out = assistant.respond(conn, users[item["role"]], item["question"])
        latencies.append((time.perf_counter() - start) * 1000)
        if len(assistant.llm_log) > calls_before:
            LLM_LOG.append(assistant.llm_log[-1])
            time.sleep(6)                                   # stay under free-tier rate limits
        trace = getattr(assistant, "last_trace", {}) if out["route"] == "retrieval" else {}
        ok, details = judge(item, out, trace)
        rows.append({"item": item, "out": out, "trace": trace, "ok": ok, **details})
    return rows, latencies


def summarise(rows, latencies):
    n = len(rows)
    pct = lambda a, b: f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a"
    answer_rows = [r for r in rows if r["item"]["type"] == "answer"]
    # retrieval metrics on answerable questions, computed from the ranked chunks of the pipeline
    recall = mrr = prec = 0
    for r in answer_rows:
        retrieved = r["trace"].get("retrieved", [])[:5]
        expected = set(r["item"]["docs"])
        ranks = [i + 1 for i, d in enumerate(retrieved) if d in expected]
        recall += bool(ranks)
        mrr += 1 / ranks[0] if ranks else 0
        prec += (sum(d in expected for d in retrieved) / len(retrieved)) if retrieved else 0
    answered = [r for r in answer_rows if r["out"]["outcome"] == "answered"]
    summary = {
        "items": n,
        "behaviour_matched": pct(sum(r["ok"] for r in rows), n),
        "retrieval_recall@5": pct(recall, len(answer_rows)),
        "MRR@5": round(mrr / len(answer_rows), 3) if answer_rows else None,
        "precision@5": round(prec / len(answer_rows), 3) if answer_rows else None,
        "required_facts_present": pct(sum(r["facts_ok"] for r in answered), len(answered)),
        "answers_citing_expected_source": pct(sum(bool(set(r["cited"]) & set(r["item"]["docs"])) for r in answered), len(answered)),
        "answers_citing_forbidden_source": sum(r["forbidden_hit"] for r in rows),
        "unanswerable_correctly_not_answered": pct(sum(r["ok"] for r in rows if r["item"]["type"] == "no_answer"),
                                                    sum(r["item"]["type"] == "no_answer" for r in rows)),
        "refusals_and_injection_handled": pct(sum(r["ok"] for r in rows if r["item"]["type"] == "refuse"),
                                               sum(r["item"]["type"] == "refuse" for r in rows)),
        "tool_and_access_items_correct": pct(sum(r["ok"] for r in rows if r["item"]["type"] in ("tool", "access")),
                                              sum(r["item"]["type"] in ("tool", "access") for r in rows)),
        "human_handoff_correct": pct(sum(r["ok"] for r in rows if r["item"]["type"] == "human"),
                                      sum(r["item"]["type"] == "human" for r in rows)),
        "latency_ms_p50_p95": f"{statistics.median(latencies):.0f} / {sorted(latencies)[int(0.95 * (len(latencies) - 1))]:.0f}",
    }
    paraphrase = [r for r in rows if r["item"]["kind"] == "paraphrase"]
    if paraphrase:
        summary["paraphrase_items_correct"] = pct(sum(r["ok"] for r in paraphrase), len(paraphrase))
    return summary


def tune():
    best = None
    grid = itertools.product([0.15, 0.20, 0.25, 0.30], [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
                             [1.5, 2.0, 3.0, 4.0, 5.0], [2, 3])
    for spec_cos, cos, mass, matched in grid:
        config.SPECIFIC_COSINE, config.MIN_COSINE, config.MIN_MASS, config.MIN_MATCHED = spec_cos, cos, mass, matched
        rows, _ = run("dev")
        score = (sum(r["ok"] for r in rows), cos, spec_cos)   # most correct first; ties go to the stricter thresholds
        if best is None or score > best[0]:
            best = (score, spec_cos, cos, mass, matched, len(rows))
    print(f"best on dev: {best[0][0]}/{best[5]} correct with SPECIFIC_COSINE={best[1]}, MIN_COSINE={best[2]}, "
          f"MIN_MASS={best[3]}, MIN_MATCHED={best[4]}")


if __name__ == "__main__":
    if "--tune" in sys.argv:
        tune()
        sys.exit(0)
    report = {}
    for split in ("dev", "test", None):
        rows, lat = run(split)
        report[split or "all"] = summarise(rows, lat)
    for name, summary in report.items():
        print(f"\n=== {name} ===")
        for k, v in summary.items():
            print(f"{k:40} {v}")
    rows, _ = run()
    if USE_LLM:
        from collections import Counter
        print("\nGemini rewrite outcomes (all runs above):", dict(Counter(x.split(":")[0] for x in LLM_LOG)))
        print("Rejected reasons:", dict(Counter(x for x in LLM_LOG if x.startswith("rejected"))))
    failures = [r for r in rows if not r["ok"]]
    print(f"\n{len(failures)} mismatches:")
    for r in failures:
        it = r["item"]
        print(f" [{it['split']}] {it['id']} ({it['type']}/{it['kind']}) {it['question']!r} -> {r['out']['route']}/{r['out']['outcome']}"
              f" cited={r['cited']} trace={ {k: round(v, 2) if isinstance(v, float) else v for k, v in r['trace'].items() if k != 'retrieved'} }")
    with open(Path(__file__).parent / ("results_llm.json" if USE_LLM else "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
