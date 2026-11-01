"""Load corpus, build all retrieval components, and expose a unified API."""
from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

from .preprocessing import preprocess, stem_tokens
from .indexing import build_inverted_index
from .boolean_retrieval import boolean_search
from .tfidf_retrieval import TfidfRetriever
from .bm25 import BM25Retriever

REQUIRED_COLUMNS = ["doc_id", "title", "url", "content", "crawl_date"]


class SearchEngine:
    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path, keep_default_na=False)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"CSV lacks required columns: {sorted(missing)}")
        df = df[REQUIRED_COLUMNS + [c for c in df.columns if c not in REQUIRED_COLUMNS]].copy()
        for col in ("doc_id", "title", "url", "content", "crawl_date"):
            df[col] = df[col].fillna("").astype(str).map(lambda x: x.replace("\ufffd", " ").strip())
        df = df[(df.doc_id != "") & (df.title != "") & (df.url != "") & (df.content.str.strip() != "")]
        df = df.drop_duplicates(subset="url", keep="first").drop_duplicates(subset="doc_id", keep="first")
        df = df.drop_duplicates(subset=["title", "content"], keep="first")
        self.raw_df = df.reset_index(drop=True)
        self.documents = []
        self.preprocessing_df = self.raw_df.copy()
        token_strings = []
        unstemmed_lists = []
        for row in self.raw_df.to_dict(orient="records"):
            # Title is repeated twice to give it a modest, transparent ranking boost.
            retrieval_text = f"{row['title']} {row['title']} {row['content']}"
            tokens = preprocess(retrieval_text, stem=False)
            unstemmed_lists.append(tokens)
        # Stem unique surface terms once; Sastrawi's word-level stemmer is much
        # faster than processing full documents repeatedly.
        vocabulary = {token for tokens in unstemmed_lists for token in tokens}
        stem_map = dict(zip(vocabulary, stem_tokens(list(vocabulary))))
        for row, raw_tokens in zip(self.raw_df.to_dict(orient="records"), unstemmed_lists):
            tokens = [stem_map[token] for token in raw_tokens]
            tokens = [token for token in tokens if token]
            doc = {**row, "tokens": tokens}
            self.documents.append(doc)
            token_strings.append(" ".join(tokens))
        self.preprocessing_df["preprocessed_text"] = token_strings
        self.index, self.term_frequencies, self.index_stats = build_inverted_index(self.documents)
        self.df = self.index_stats["df"]
        self.all_doc_ids = set(self.raw_df.doc_id)
        self.tfidf = TfidfRetriever(self.documents, self.index, self.df, preprocess)
        self.bm25 = BM25Retriever(self.documents, self.index, self.df, preprocess)
        self.doc_by_id = {d["doc_id"]: d for d in self.documents}

    def search(self, query: str, model: str = "bm25", top_k: int = 10) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized_model = model.lower().replace(" ", "_")
        if normalized_model in {"tfidf", "tf_idf", "cosine", "tfidf_cosine", "tf_idf_cosine"}:
            return self.tfidf.search(query, top_k)
        if normalized_model == "bm25":
            return self.bm25.search(query, top_k)
        raise ValueError("model must be 'tfidf' or 'bm25'")

    def boolean(self, query: str) -> list[dict]:
        def normalize_term(term: str) -> str:
            tokens = preprocess(term)
            return tokens[0] if tokens else ""
        ids = boolean_search(query, self.index, self.all_doc_ids, normalize_term)
        return [{"doc_id": doc_id, "title": self.doc_by_id[doc_id]["title"], "url": self.doc_by_id[doc_id]["url"]}
                for doc_id in sorted(ids)]

    def save_artifacts(self, preprocessed_path: str | Path, index_path: str | Path) -> None:
        Path(preprocessed_path).parent.mkdir(parents=True, exist_ok=True)
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        self.preprocessing_df.to_csv(preprocessed_path, index=False, encoding="utf-8-sig")
        payload = {"n_documents": len(self.documents), "df": self.df, "postings": self.index,
                   "doc_lengths": self.index_stats["doc_lengths"]}
        Path(index_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def display_results(results: list[dict]) -> None:
    if not results:
        print("Tidak ditemukan dokumen yang cocok dengan query.")
        return
    for result in results:
        print(f"{result['rank']}. {result['title']} [{result['score']:.4f}]")
        print(f"   {result['url']}")
        print(f"   {result['snippet']}\n")
