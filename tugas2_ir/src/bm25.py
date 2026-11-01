"""BM25 implementation with Robertson-Sparck Jones IDF and length normalization."""
from __future__ import annotations

import math
from collections import Counter


class BM25Retriever:
    def __init__(self, documents: list[dict], index: dict, df: dict[str, int], preprocess, k1: float = 1.2, b: float = 0.75):
        self.documents = {d["doc_id"]: d for d in documents}
        self.index = index
        self.df = df
        self.preprocess = preprocess
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.lengths = {d["doc_id"]: len(d["tokens"]) for d in documents}
        self.avgdl = sum(self.lengths.values()) / max(self.n_docs, 1)
        self.tf = {d["doc_id"]: Counter(d["tokens"]) for d in documents}

    def term_idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Positive BM25 IDF; smoothing keeps ubiquitous terms finite.
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_document(self, query_terms: list[str], doc_id: str) -> float:
        score = 0.0
        dl = self.lengths[doc_id]
        for term in query_terms:
            freq = self.tf[doc_id].get(term, 0)
            if not freq:
                continue
            norm = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-12))
            score += self.term_idf(term) * (freq * (self.k1 + 1.0) / norm)
        return score

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_terms = self.preprocess(query)
        candidates = set()
        for term in query_terms:
            candidates.update(p["doc_id"] for p in self.index.get(term, []))
        scored = [(self.score_document(query_terms, doc_id), doc_id) for doc_id in candidates]
        scored = [(score, doc_id) for score, doc_id in scored if score > 0]
        scored.sort(key=lambda x: (-x[0], x[1]))
        results = []
        for rank, (score, doc_id) in enumerate(scored[:top_k], 1):
            doc = self.documents[doc_id]
            from .tfidf_retrieval import make_snippet
            results.append({"rank": rank, "doc_id": doc_id, "title": doc["title"], "url": doc["url"],
                            "score": score, "snippet": make_snippet(doc["content"], query_terms)})
        return results
