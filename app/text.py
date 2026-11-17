"""Text processing: tokenising, light stemming, and the everyday-wording layer."""
import re

STOPWORDS = set(
    "a an the of to in on for and or is are was were be been it its this that these those "
    "i me my we our you your do does did can could should would will shall how what when where "
    "which who whom why with as at by from if about into than then so there their they them "
    "have has had am just also get got".split()
)

# Everyday phrases -> the terms the documents actually use. The expansion is
# appended to the query; the original wording is kept as well.
WORDING_RULES = [
    (r"\b(reject|rejected|refus\w*|turned down|declin\w*|not paid|wouldn'?t pay|won'?t pay)\b", "claim denied appeal"),
    (r"\b(dispute|contest|challenge|fight)\b", "appeal deadline"),
    (r"how long (do|can|will) (i|we|you)\b.{0,15}\bto (appeal|file|dispute|contest)", "appeal deadline calendar days filed"),
    (r"before (my |the )?(insurance|plan|insurer) (starts?|pays?|kicks? in)|owe before|pay (first|before)", "deductible"),
    (r"(most|max(imum)?) (i|you)('| a)?(ll| will| would| could| can)? (pay|spend)|spend in a year", "out-of-pocket maximum"),
    (r"^(?!.*(online|virtual|video|tele)).*(\b(doctor|gp|physician) (visit|appointment)s?\b|see (my |a )?(doctor|gp|physician)|cost to see)", "primary care visit copay"),
    (r"\b(er|a&e)\b|emergency", "emergency room copay"),
    (r"\burgent care\b|\bwalk-?in\b", "urgent care copay"),
    (r"pre-?auth\w*|pre-?approval|approval (before|for)|permission (before|for)|authori[sz]ation", "prior authorization"),
    (r"\b(medicine|medication|drug|pill|prescription)s? (cost|price)|generic", "prescription generic copay"),
    (r"mail[- ]?order|home delivery|90[- ]?day", "mail-order pharmacy 90-day supply"),
    (r"telehealth|telemedicine|online doctor|video (visit|call)|virtual", "virtual visit"),
    (r"^(?!.*(appeal|den(y|ied|ial)|dispute|reject)).*(how long.*(claim|process)|claim.*(take|long))", "claim processing days"),
    (r"\bshare\b.*\bplan pays?\b|plan pays? .*(after|once)", "coinsurance plan pays percent"),
    (r"\bscan\b|\bmri\b|\bct\b|imaging", "advanced imaging mri coinsurance"),
    (r"\bbill\b", "explanation of benefits eob billed"),
]

_SUFFIXES = ("ations", "ation", "ments", "ment", "ings", "ing", "edly", "ied", "ies", "ed", "es", "s")


def stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf in ("ies", "ied"):
                return word[: -len(suf)] + "y"
            return word[: -len(suf)]
    return word


def normalise(text: str) -> str:
    text = text.lower().replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip()


def expand_wording(query: str):
    """Return (expanded_query, list_of_expansions_applied)."""
    q = normalise(query)
    added = []
    for pattern, expansion in WORDING_RULES:
        if re.search(pattern, q):
            added.append(expansion)
    return (q + " " + " ".join(added)).strip(), added


def tokens(text: str, keep_stopwords: bool = False):
    words = re.findall(r"[a-z0-9$%][a-z0-9,.\-$%]*[a-z0-9%]|[a-z0-9]", normalise(text))
    out = []
    for w in words:
        w = w.strip(".,-")
        if not w:
            continue
        if not keep_stopwords and w in STOPWORDS:
            continue
        out.append(stem(w))
    return out


def split_sentences(text: str):
    sentences = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*[-*]\s+", "", line).strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z$0-9])", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences
