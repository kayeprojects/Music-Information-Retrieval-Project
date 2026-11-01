"""TF-IDF Vector Space Model and cosine ranking, implemented explicitly."""
from __future__ import annotations

import math
from collections import Counter


def tf_weight(tf: int) -> float:
    """Log-normalized TF: 1 + log(tf), zero when term is absent."""
    return 1.0 + math.log(tf) if tf > 0 else 0.0


def idf_weight(n_docs: int, df: int) -> float:
    """Smoothed IDF: log((N + 1)/(df + 1)) + 1; avoids zero/undefined values."""
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def vectorize(counter: Counter, idf: dict[str, float]) -> dict[str, float]:
    return {term: tf_weight(tf) * idf.get(term, 0.0) for term, tf in counter.items() if term in idf}


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class TfidfRetriever:
    def __init__(self, documents: list[dict], index: dict, df: dict[str, int], preprocess):
        self.documents = {d["doc_id"]: d for d in documents}
        self.index = index
        self.df = df
        self.preprocess = preprocess
        self.n_docs = len(documents)
        self.idf = {term: idf_weight(self.n_docs, freq) for term, freq in df.items()}
        self.doc_vectors = {}
        for doc in documents:
            counts = Counter(doc["tokens"])
            self.doc_vectors[doc["doc_id"]] = vectorize(counts, self.idf)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_tokens = self.preprocess(query)
        query_vector = vectorize(Counter(query_tokens), self.idf)
        candidate_ids = set()
        for term in query_vector:
            candidate_ids.update(p["doc_id"] for p in self.index.get(term, []))
        scored = []
        for doc_id in candidate_ids:
            score = cosine_similarity(query_vector, self.doc_vectors[doc_id])
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._result(doc_id, score, query_tokens, rank) for rank, (score, doc_id) in enumerate(scored[:top_k], 1)]

    def _result(self, doc_id: str, score: float, query_tokens: list[str], rank: int) -> dict:
        doc = self.documents[doc_id]
        return {"rank": rank, "doc_id": doc_id, "title": doc["title"], "url": doc["url"], "score": score,
                "snippet": make_snippet(doc["content"], query_tokens)}


def make_snippet(content: str, terms: list[str], length: int = 260) -> str:
    text = " ".join(str(content).split())
    lowered = text.casefold()
    positions = [lowered.find(term.casefold()) for term in terms if term and lowered.find(term.casefold()) >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    snippet = text[start:start + length]
    return ("…" if start else "") + snippet + ("…" if start + length < len(text) else "")
