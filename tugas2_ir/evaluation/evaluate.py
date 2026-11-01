"""Evaluate TF-IDF/cosine and BM25 against transparent domain criteria."""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from src.search_engine import SearchEngine

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset_crawling_ir.csv"
QUERY_FILE = ROOT / "evaluation" / "queries.csv"
GT_FILE = ROOT / "evaluation" / "ground_truth.csv"
RESULT_FILE = ROOT / "evaluation" / "metrics.csv"
DETAIL_FILE = ROOT / "evaluation" / "ranking_comparison.csv"


def criterion_terms(raw: str) -> list[str]:
    return [x.strip().casefold() for x in str(raw).split("|") if x.strip()]


def matching_terms(text: str, terms: list[str]) -> list[str]:
    """Match whole words/phrases so e.g. 'band' does not match 'bandara'."""
    text = text.casefold()
    hits = []
    for term in terms:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        if re.search(pattern, text):
            hits.append(term)
    return hits


def label_ground_truth(engine: SearchEngine, query_df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based manual-reviewable labels, based on title + opening 900 chars."""
    labels = []
    for row in query_df.to_dict(orient="records"):
        terms = criterion_terms(row["ground_truth_terms"])
        for doc in engine.documents:
            title = doc["title"].casefold()
            lead = doc["content"][:900].casefold()
            title_hits = matching_terms(title, terms)
            lead_hits = matching_terms(lead, terms)
            # Titles provide primary evidence; lead-only matches need multiple signals.
            relevant = bool(title_hits) or len(set(lead_hits)) >= 2
            if relevant:
                labels.append({"query_id": row["query_id"], "doc_id": doc["doc_id"],
                               "label": 1, "evidence": "title: " + "|".join(title_hits) if title_hits else "lead: " + "|".join(sorted(set(lead_hits)))})
    return pd.DataFrame(labels, columns=["query_id", "doc_id", "label", "evidence"])


def metrics_at_k(retrieved: list[str], relevant: set[str], k: int) -> dict:
    top = retrieved[:k]
    hits = sum(doc_id in relevant for doc_id in top)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": hits}


def main() -> None:
    engine = SearchEngine(DATA)
    queries = pd.read_csv(QUERY_FILE, keep_default_na=False)
    ground_truth = label_ground_truth(engine, queries)
    ground_truth.to_csv(GT_FILE, index=False, encoding="utf-8-sig")
    gt_map = {qid: set(group.doc_id) for qid, group in ground_truth.groupby("query_id")}
    metric_rows, comparison_rows = [], []
    for query in queries.to_dict(orient="records"):
        qid, text = query["query_id"], query["query"]
        relevant = gt_map.get(qid, set())
        model_results = {}
        for model in ("tfidf", "bm25"):
            results = engine.search(text, model=model, top_k=10)
            ids = [r["doc_id"] for r in results]
            model_results[model] = results
            scores = metrics_at_k(ids, relevant, 10)
            metric_rows.append({"query_id": qid, "query": text, "model": model, "ground_truth_count": len(relevant),
                                **{f"{key}@10": value for key, value in scores.items()}})
        cosine_ids = [r["doc_id"] for r in model_results["tfidf"]]
        bm25_ids = [r["doc_id"] for r in model_results["bm25"]]
        overlap = len(set(cosine_ids[:10]) & set(bm25_ids[:10]))
        comparison_rows.append({"query_id": qid, "query": text, "ground_truth_count": len(relevant),
                                "top10_overlap": overlap, "top10_jaccard": overlap / max(len(set(cosine_ids[:10]) | set(bm25_ids[:10])), 1),
                                "tfidf_top5": "|".join(cosine_ids[:5]), "bm25_top5": "|".join(bm25_ids[:5])})
    metrics = pd.DataFrame(metric_rows)
    comparison = pd.DataFrame(comparison_rows)
    metrics.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    comparison.to_csv(DETAIL_FILE, index=False, encoding="utf-8-sig")
    print(f"Corpus documents: {len(engine.documents)}")
    print(f"Vocabulary terms: {len(engine.index)}")
    print(f"Ground-truth query count: {queries.query_id.nunique()}")
    print(f"Ground-truth labels: {len(ground_truth)}")
    print("\nMean metrics @10 by model:")
    print(metrics.groupby("model")[["precision@10", "recall@10", "f1@10"]].mean().round(4))
    print("\nPer-query model comparison:")
    print(comparison[["query_id", "top10_overlap", "top10_jaccard"]].to_string(index=False))
    print(f"\nSaved: {GT_FILE.relative_to(ROOT)}, {RESULT_FILE.relative_to(ROOT)}, {DETAIL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
