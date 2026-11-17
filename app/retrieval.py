"""Permission-aware hybrid retrieval.

1. Chunks the user may not read are removed BEFORE any ranking, and the ranking
   statistics (document frequencies) are computed only from the chunks that remain,
   so a document the user cannot see cannot influence their answer.
2. Two independent rankers score the remaining chunks: BM25 (keyword) and TF-IDF
   cosine similarity over unigrams and bigrams (vector).
3. The two rankings are merged with Reciprocal Rank Fusion: score = sum 1 / (k + rank).
"""
import json
import math
import re
from collections import Counter
from . import config
from .auth import DOC_AUDIENCES
from .text import expand_wording, split_sentences, tokens

BM25_K1 = 1.5
BM25_B = 0.75


def load_chunks(manifest_path=None, docs_dir=None):
    manifest_path = manifest_path or config.DATA_DIR / "manifest.json"
    docs_dir = docs_dir or config.DOCS_DIR
    chunks = []
    with open(manifest_path, encoding="utf-8") as f:
        documents = json.load(f)["documents"]
    for doc in documents:
        text = open(docs_dir / doc["file"], encoding="utf-8").read()
        parts = re.split(r"^## ", text, flags=re.M)[1:]   # text before the first '## ' is the intro; it is skipped
        for n, part in enumerate(parts):
            heading, _, body = part.partition("\n")
            chunks.append({
                "id": f"{doc['id']}#{n}", "doc_id": doc["id"], "title": doc["title"],
                "section": heading.strip(), "text": body.strip(), "audience": doc["audience"],
                "plan": doc["plan"], "version": doc["version"], "superseded": doc["superseded"],
            })
    return chunks


def _bigrams(toks):
    return [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


class Retriever:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else load_chunks()
        for c in self.chunks:
            feature_text = f"{c['title']} {c['section']} {c['section']} {c['text']}"
            c["_tokens"] = tokens(feature_text)

    # -- permission filter -------------------------------------------------
    def readable(self, user):
        audiences = DOC_AUDIENCES.get(user["role"], set())
        plan = user.get("plan")

        def ok(c):
            if c["superseded"] or c["audience"] not in audiences:
                return False
            if user["role"] == "member" and c["plan"] not in (None, plan):
                return False
            return True
        return [c for c in self.chunks if ok(c)]

    # -- ranking ------------------------------------------------------------
    def search(self, query, user, top_k=None):
        top_k = top_k or config.TOP_K
        candidates = self.readable(user)
        expanded, added = expand_wording(query)
        original_tokens = tokens(query)
        extra_tokens = tokens(" ".join(added))
        q_terms = original_tokens + [t for t in extra_tokens if t not in original_tokens]
        info = {"expanded": added, "q_terms": q_terms, "hits": [], "idf": {}, "coverage": 0.0,
                "matched_mass": 0.0, "n_matched": 0, "agree": False}
        if not candidates or not q_terms:
            return info

        n = len(candidates)
        df = Counter()
        bi_df = Counter()
        for c in candidates:
            df.update(set(c["_tokens"]))
            bi_df.update(set(_bigrams(c["_tokens"])))
        avgdl = sum(len(c["_tokens"]) for c in candidates) / n

        def idf(term):
            d = df.get(term, 0)
            return math.log(1 + (n - d + 0.5) / (d + 0.5))

        def bi_idf(term):
            d = bi_df.get(term, 0)
            return math.log(1 + (n - d + 0.5) / (d + 0.5))

        q_tf = Counter(original_tokens * 2 + extra_tokens)       # original wording counts double
        q_big = Counter(_bigrams(original_tokens))

        def vector(toks):
            tf = Counter(toks)
            vec = {t: (1 + math.log(f)) * idf(t) for t, f in tf.items()}
            for b, f in Counter(_bigrams(toks)).items():
                vec[b] = (1 + math.log(f)) * bi_idf(b)
            return vec

        q_vec = {t: (1 + math.log(f)) * idf(t) for t, f in q_tf.items()}
        for b, f in q_big.items():
            q_vec[b] = (1 + math.log(f)) * bi_idf(b)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored = []
        for c in candidates:
            tf = Counter(c["_tokens"])
            dl = len(c["_tokens"])
            bm25 = 0.0
            for t in set(q_tf):
                if t in tf:
                    f = tf[t]
                    bm25 += idf(t) * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl))
            vec = vector(c["_tokens"])
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            cos = sum(w * vec.get(t, 0.0) for t, w in q_vec.items()) / (q_norm * norm)
            scored.append({"chunk": c, "bm25": bm25, "cosine": cos})

        for key in ("bm25", "cosine"):
            ordered = sorted((s for s in scored if s[key] > 0), key=lambda s: -s[key])
            for rank, s in enumerate(ordered, 1):
                s[f"rank_{key}"] = rank
        for s in scored:
            s["rrf"] = sum(1.0 / (config.RRF_K + s[r]) for r in ("rank_bm25", "rank_cosine") if r in s)
        hits = sorted((s for s in scored if s["rrf"] > 0), key=lambda s: -s["rrf"])[:top_k]

        # coverage: share of the question's term weight that the best chunk actually contains
        total = sum(idf(t) for t in set(q_terms))
        if hits and total:
            present = set(hits[0]["chunk"]["_tokens"])
            matched = [t for t in set(q_terms) if t in present]
            info["coverage"] = sum(idf(t) for t in matched) / total
            info["matched_mass"] = sum(idf(t) for t in matched)
            info["n_matched"] = len(matched)
            info["agree"] = hits[0].get("rank_bm25") == 1 and hits[0].get("rank_cosine") == 1
        info.update({"hits": hits, "idf": idf, "q_vec": q_vec, "q_norm": q_norm, "vector": vector})
        return info

    # -- sentence extraction ------------------------------------------------
    def best_sentences(self, info, max_sentences=None, pool=3):
        """Pick the sentences most similar to the question from the top chunks."""
        max_sentences = max_sentences or config.MAX_SENTENCES
        if not info["hits"]:
            return []
        q_vec, q_norm, vector = info["q_vec"], info["q_norm"], info["vector"]
        picked = []
        for rank, hit in enumerate(info["hits"][:pool]):
            for pos, sentence in enumerate(split_sentences(hit["chunk"]["text"])):
                toks = tokens(sentence)
                if not toks:
                    continue
                toks = toks + tokens(f"{hit['chunk']['section']} {hit['chunk']['title']}")   # headings give context
                vec = vector(toks)
                norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
                cos = sum(w * vec.get(t, 0.0) for t, w in q_vec.items()) / (q_norm * norm)
                picked.append({"sentence": sentence, "cosine": cos, "chunk": hit["chunk"], "rank": rank, "pos": pos})
        picked.sort(key=lambda p: -p["cosine"])
        chosen = [p for p in picked[:max_sentences] if p["cosine"] > 0]
        chosen.sort(key=lambda p: (p["rank"], p["pos"]))
        return chosen
