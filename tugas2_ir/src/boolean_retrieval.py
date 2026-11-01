"""Basic AND/OR/NOT Boolean retrieval over the inverted index."""
from __future__ import annotations

import re
from .indexing import postings_doc_ids

BOOL_TOKEN_RE = re.compile(r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[^\s()]+", re.IGNORECASE)


def boolean_search(query: str, index: dict, all_doc_ids: set[str], normalize_term=lambda x: x) -> set[str]:
    """Evaluate a Boolean expression; AND binds tighter than OR; NOT is unary."""
    raw_tokens = BOOL_TOKEN_RE.findall(query)
    if not raw_tokens:
        return set()
    tokens = [t.upper() if t.upper() in {"AND", "OR", "NOT"} or t in "()" else normalize_term(t) for t in raw_tokens]
    position = 0

    def parse_atom() -> set[str]:
        nonlocal position
        if position >= len(tokens):
            return set()
        token = tokens[position]
        if token == "NOT":
            position += 1
            return all_doc_ids - parse_atom()
        if token == "(":
            position += 1
            result = parse_or()
            if position < len(tokens) and tokens[position] == ")":
                position += 1
            return result
        if token in {"AND", "OR", ")"}:
            position += 1
            return set()
        position += 1
        return postings_doc_ids(index, token)

    def parse_and() -> set[str]:
        nonlocal position
        result = parse_atom()
        while position < len(tokens) and tokens[position] == "AND":
            position += 1
            result &= parse_atom()
        return result

    def parse_or() -> set[str]:
        nonlocal position
        result = parse_and()
        while position < len(tokens) and tokens[position] == "OR":
            position += 1
            result |= parse_and()
        return result

    return parse_or()
