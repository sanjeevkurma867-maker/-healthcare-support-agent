"""Deterministic intent router. It runs BEFORE retrieval, so unsafe requests never reach the answer step."""
import re

INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\b.{0,30}\b(instructions?|rules?|guidelines?|prompt)"
    r"|\b(system|hidden|developer) (prompt|message|instructions?)\b"
    r"|reveal.{0,20}(instructions?|prompt|rules)"
    r"|\b(developer|admin|debug|jailbreak|dan) mode\b|\byou are now\b|pretend (you are|to be)"
    r"|act as (an? )?(admin|administrator|developer)", re.I)

MEDICAL = re.compile(
    r"\bshould i (take|stop|use|start|skip|increase|reduce)\b|\bdos(e|age)s?\b|\bhow many (pills|tablets|mg)\b"
    r"|\bdiagnos\w*|\bsymptoms?\b|\bside[- ]effects?\b|\bis it safe to\b|\bcan i (take|mix|combine)\b.{0,30}\bwith\b"
    r"|\b(what|which) (medicine|medication|drug|treatment)\b.{0,20}\b(should|can|do|for)\b"
    r"|\bdo i have (cancer|diabetes|an infection|a \w+ disease)\b|\bprescribe\b|\bwhat'?s wrong with me\b", re.I)

DECISION = re.compile(
    r"\b(approve|deny|overturn|overrule|reverse)\b.{0,25}\b(my|this|the|that|a)\b.{0,25}"
    r"\b(claim|request|authori[sz]ation|prior auth\w*|appeal|surgery)\b"
    r"|\bwill (my|this|the) (claim|request|surgery|authori[sz]ation|procedure)\w* be (approved|covered|paid|denied)\b"
    r"|\bwill you (approve|pay|cover)\b|\bcan you (approve|pay)\b|\bguarantee (coverage|approval)\b", re.I)

HUMAN = re.compile(
    r"\b(talk|speak|chat|connect|transfer)\b.{0,15}\b(human|person|representative|agent|someone|advisor|staff)\b"
    r"|\b(human agent|real person|live agent|customer (service|care)|support (representative|agent|team))\b"
    r"|\b(create|open|raise|log) (a |an )?(support )?(case|ticket|request)\b", re.I)

OTHER_MEMBER = re.compile(r"SYN-MBR-\d+", re.I)
OTHER_PERSON_DATA = re.compile(
    r"\b(another|other|someone else'?s?|everyone'?s|all|every)\b.{0,20}\b(members?|patients?|customers?|users?|people)\b.{0,25}\b(claims?|records?|data|info\w*|plans?)\b"
    r"|\b(every|all)\b.{0,15}\b(claims?|records?)\b.{0,20}\b(database|system)\b"
    r"|\b(print|dump|export)\b.{0,25}\b(every|all)\b.{0,20}\b(claims?|records?|members?)\b")
POSSESSIVE_NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)?'s\s+(claims?|records?|data|plan)\b")

GREETING = re.compile(r"^\W*(hi|hii+|hello|hey|good (morning|afternoon|evening)|thanks|thank you|ok|okay|help|what can you do)\W*$", re.I)
CLAIM_NUMBER = re.compile(r"\bSYN-CLM-\d{4}-\d{6}\b", re.I)
LIST_CLAIMS = re.compile(r"^\W*claims?\W*$|\bmy\b.{0,15}\bclaims?\b|\bclaims? status\b|\b(show|list|view|display|see|check)\b.{0,15}\bclaims?\b", re.I)
CLAIMS_EXCLUDE = re.compile(r"\b(how|why|what happens|appeal|den(y|ied|ial)|reject\w*|dispute|deadline|limit|take|long|mean|time)\b", re.I)
PLAN_OWN = re.compile(r"\b(my|our)\b.{0,20}\b(deductible|co-?pay|coinsurance|out.of.pocket|benefits?|plan|premium)\b", re.I)
PLAN_STAFF = re.compile(r"\bbenefits?\b.{0,20}\b(gold|silver)\b|\b(gold|silver) (plan )?(benefits?|deductible|co-?pay)\b", re.I)
REQUIRED_DOCS = re.compile(
    r"\b(documents?|paperwork|requirements?|checklist|packet)\b.{0,60}(prior auth\w*|authori[sz]ation|pre-?auth)"
    r"|(prior auth\w*|authori[sz]ation|pre-?auth).{0,60}\b(documents?|paperwork|requirements?|checklist|packet)\b"
    r"|\bwhat (do|does|should) (i|we|a request|the request) (need|include|submit)\b.{0,60}(prior auth\w*|authori[sz]ation)", re.I)

FOCUS = [("deductible", "deductible"), ("out", "oop_max"), ("specialist", "specialist_copay"), ("primary", "pcp_copay"),
         ("urgent", "urgent_copay"), ("er ", "er_copay"), ("emergency", "er_copay"), ("generic", "generic_rx_copay"),
         ("brand", "brand_rx_copay"), ("virtual", "virtual_copay"), ("coinsurance", "coinsurance_pct")]


def route(message: str, user: dict) -> dict:
    text = (message or "").strip()
    if not text:
        return {"kind": "empty"}
    if GREETING.match(text):
        return {"kind": "greeting"}
    if INJECTION.search(text):
        return {"kind": "injection"}
    own = (user.get("member_number") or "").upper()
    others = [m for m in OTHER_MEMBER.findall(text) if m.upper() != own]
    if others or OTHER_PERSON_DATA.search(text) or POSSESSIVE_NAME.search(text):
        return {"kind": "refuse_privacy"}
    if MEDICAL.search(text):
        return {"kind": "refuse_medical"}
    if DECISION.search(text):
        return {"kind": "refuse_decision"}
    if user["role"] in ("member", "provider") and HUMAN.search(text):
        return {"kind": "human"}
    m = CLAIM_NUMBER.search(text)
    if m:
        return {"kind": "tool", "tool": "claim_status", "args": {"claim_number": m.group(0).upper()}}
    low = text.lower() + " "
    if user["role"] == "member" and LIST_CLAIMS.search(text) and not CLAIMS_EXCLUDE.search(text):
        return {"kind": "tool", "tool": "list_my_claims", "args": {}}
    focus = next((field for key, field in FOCUS if key in low), None)
    if user["role"] == "member" and PLAN_OWN.search(text):
        return {"kind": "tool", "tool": "plan_benefits", "args": {}, "focus": focus}
    sm = PLAN_STAFF.search(text)
    if user["role"] != "member" and sm:
        code = "GOLD" if "gold" in low else "SILVER"
        return {"kind": "tool", "tool": "plan_benefits", "args": {"plan_code": code}, "focus": focus}
    if REQUIRED_DOCS.search(text):
        return {"kind": "tool", "tool": "required_documents", "args": {"procedure": text[:200]}, "fallback": "retrieval"}
    return {"kind": "retrieval"}
