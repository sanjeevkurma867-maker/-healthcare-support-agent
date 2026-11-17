# Healthcare Support Agent (synthetic prototype)

A small, working conversational assistant for a healthcare support portal. Members, providers, support agents
and administrators sign in and see screens for their role. The assistant answers questions about plan benefits,
claims and prior authorization **only from verified sources**, shows where each answer came from, refuses medical
advice, never approves or denies anything, and hands unresolved questions to a person.

> **All data is fictional.** This is an independent academic case study. It is not an Evernorth Health Services
> product and uses no Evernorth systems, data or confidential information.

## What it does

| Capability | How it works |
|---|---|
| Grounded answers with citations | Permission-filtered hybrid retrieval (BM25 + TF-IDF cosine, merged with Reciprocal Rank Fusion), an evidence check, then quoted sentences with `[C1]` markers that are validated against the retrieved sources |
| "I won't guess" | If the evidence is too weak, the assistant says so and offers a support case |
| Everyday wording | A rule layer maps phrases such as "they rejected my claim" to the documents' terms ("denied", "appeal") |
| Safe refusals | A deterministic router runs before retrieval and refuses medical advice, approval/denial requests, other people's data and prompt injection |
| Controlled tool use | Four allow-listed tools (`claim_status`, `list_my_claims`, `plan_benefits`, `required_documents`), each schema-checked, permission-checked, call-budgeted and audited |
| Role-based access | Four roles and a single permission map; documents are filtered in code **before** ranking, so a document a user may not read cannot influence their answer |
| Human escalation | The assistant proposes a case with a summary; it is created only after the user confirms, once per idempotency key; cases follow a checked lifecycle |
| Audit and admin | Append-only audit log (database triggers), metrics, knowledge gaps (questions the assistant could not answer) |
| Security basics | PBKDF2 password hashing, signed expiring tokens that can be revoked at sign-out, login rate limiting, same error for unknown email and wrong password |

## Run it

Needs only **Python 3.9 or newer**. There is nothing to install.

```bash
python run.py
```

Open http://127.0.0.1:8000 and sign in with a demo account (password `demo123`):

| Account | Role |
|---|---|
| priya@example.test | Member, Gold plan |
| ravi@example.test | Member, Silver plan |
| jordan@example.test | Provider |
| sam@example.test | Support agent |
| alex@example.test | Administrator |

The database file `hsa.db` is created and seeded on first start. Delete it to reset the demo.

Things to try as Priya: "What is my deductible?", "Show my claims", "They rejected my claim, what is the time limit to dispute it?",
"Should I take ibuprofen for my back pain?", "What is the weather today?" (then confirm the support case), then sign in as Sam to
work the case, and as Alex to see the dashboard and audit log.

## Optional: Gemini rewrite (off by default)

The assistant works without any language model. If you want more natural wording, you can switch on a Gemini rewrite.
Retrieval, permissions, refusals and tools are unchanged. Gemini only receives the question and the retrieved plan
passages (never claim records or personal data), and it only rewrites an answer that retrieval already grounded.

The rewrite is accepted only if it passes checks: every `[C1]`-style marker points at a real source, every number or
amount appears in the retrieved passages, and it contains no medical advice or approval/denial statements. If a check
fails, or the call fails, times out or has no key, the assistant uses the quoted answer instead.

```bash
# Windows PowerShell
$env:HSA_LLM="gemini"; $env:GEMINI_API_KEY="your-key"; python run.py
# Windows cmd:  set HSA_LLM=gemini  and  set GEMINI_API_KEY=your-key  then  python run.py
# macOS / Linux
HSA_LLM=gemini GEMINI_API_KEY="your-key" python run.py
```

Set `GEMINI_MODEL` to change the model (default `gemini-3.5-flash`; check Google AI Studio for the names available to
your key). **Never put the key in the code or commit it to GitHub.** The question text leaves your machine in this mode,
so use synthetic data only.

To compare both modes on the evaluation set (about 5 minutes because of rate-limit pauses):
`python eval/run_eval.py --llm`, which writes `eval/results_llm.json` and prints how many rewrites were accepted or rejected.
Rewrites reword the quoted sentences, so the evaluation's exact-phrase fact checks can fail on a correct rewrite; read the
accepted/rejected counts together with the behaviour score.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v     # 59 automated tests (the Gemini layer is tested with a fake network)
python eval/run_eval.py                      # 76-question evaluation, prints results and writes eval/results.json
python eval/run_eval.py --tune               # re-tune evidence thresholds on the dev split only
```

The evaluation set (`eval/questions.json`) has 76 synthetic questions: definitions, plan costs, appeals, claims,
prior authorization, paraphrases, tool lookups, access boundaries, medical and decision refusals, prompt injection,
questions the documents cannot answer, and requests for a person. The evidence thresholds were chosen on the
**dev** split (27 items). The **test** split (49 items) was reported, not tuned on.

| Split | Behaviour matched |
|---|---|
| Dev (27) | 100% (27/27) |
| Test (49) | 89.8% (44/49) |
| All (76) | 93.4% (71/76) |

Other results on all 76 items: retrieval recall@5 100% (45/45 answerable), MRR@5 0.989, no answer cited a superseded
or forbidden source, 8/8 unanswerable questions correctly not answered, 12/12 refusal and injection items handled,
9/9 tool and access items correct, pipeline latency about 3 ms (p50) on a laptop CPU.

## Honest limitations

- All data and questions are synthetic and small (14 documents, 76 questions). The results do not show how it would
  perform with real members, plans or documents.
- The question set was written by the project authors, and the dev/test split is not independent: one wording rule
  was added after a dev failure, and the five remaining test mismatches were left unfixed on purpose.
- Retrieval is lexical (BM25 and TF-IDF with a hand-written everyday-wording layer). It does not understand meaning,
  so unfamiliar phrasing can be missed. Paraphrase accuracy on the test split was 60% (3/5).
- Answers are extractive (quoted sentences). The assistant cannot combine facts into new sentences.
- The results above are for the default quoted-answer mode. The optional Gemini mode is tested only with a fake network in
  the automated tests; it has not been measured on real calls unless you run `python eval/run_eval.py --llm` yourself.
- There is no multilingual support.
- It runs a single local server with SQLite. It has not been security-reviewed or deployed for real use.

## Project layout

```
app/        assistant, router, retrieval, tools, cases, auth, database, server, static/index.html
data/       manifest.json, seed.json, docs/*.md (14 fictional documents)
eval/       questions.json, run_eval.py, results.json
tests/      test_core.py
```

## Ideas for future work

- Dense sentence embeddings (for example BGE-small) alongside BM25 for better paraphrase handling.
- An approved language model for natural answers, keeping the evidence check and citation validation.
- A larger, independently written evaluation set.
- Telugu and Hindi replies, expert-reviewed.
- Containerised deployment with HTTPS and monitoring after a security and privacy review.
