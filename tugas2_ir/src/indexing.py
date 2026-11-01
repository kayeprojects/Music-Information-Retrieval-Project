"""Term dictionary and postings lists with term and document frequencies."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable


def build_inverted_index(documents: Iterable[dict]) -> tuple[dict, dict, dict]:
    index: dict[str, list[dict]] = defaultdict(list)
    term_frequencies: dict[str, dict[str, int]] = {}
    doc_lengths: dict[str, int] = {}
    for doc in documents:
        doc_id = str(doc["doc_id"])
        tokens = doc["tokens"]
        counts = Counter(tokens)
        term_frequencies[doc_id] = dict(counts)
        doc_lengths[doc_id] = len(tokens)
        for term, tf in counts.items():
            index[term].append({"doc_id": doc_id, "tf": int(tf)})
    sorted_index = {term: sorted(postings, key=lambda x: x["doc_id"]) for term, postings in sorted(index.items())}
    document_frequency = {term: len(postings) for term, postings in sorted_index.items()}
    return sorted_index, term_frequencies, {"doc_lengths": doc_lengths, "df": document_frequency}


def postings_doc_ids(index: dict, term: str) -> set[str]:
    return {p["doc_id"] for p in index.get(term, [])}
