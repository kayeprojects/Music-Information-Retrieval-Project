"""Audit key Tugas 2 deliverables and print a requirement checklist."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parent
checks: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, details: str = "") -> None:
    checks.append((label, bool(condition), details))


def main() -> None:
    required = {"doc_id", "title", "url", "content", "crawl_date"}
    raw = pd.read_csv(ROOT / "data" / "dataset_crawling_ir.csv", keep_default_na=False)
    processed = pd.read_csv(ROOT / "data" / "dataset_preprocessed.csv", keep_default_na=False)
    index = json.loads((ROOT / "index" / "inverted_index.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(ROOT / "evaluation" / "queries.csv", keep_default_na=False)
    truth = pd.read_csv(ROOT / "evaluation" / "ground_truth.csv", keep_default_na=False)
    metrics = pd.read_csv(ROOT / "evaluation" / "metrics.csv", keep_default_na=False)
    comparison = pd.read_csv(ROOT / "evaluation" / "ranking_comparison.csv", keep_default_na=False)
    report_path = ROOT / "Laporan_Tugas2_IR_Musik_Indonesia.docx"
    report = Document(report_path)
    forced_breaks = sum(len(p._p.xpath('.//w:br[@w:type="page"]')) for p in report.paragraphs)

    check("Raw dataset has the five required columns", required.issubset(raw.columns), str(sorted(required)))
    check("Raw dataset contains at least 100 documents", len(raw) >= 100, f"rows={len(raw)}")
    check("Required raw fields have no empty values", not raw[list(required)].eq("").any().any(), "all required fields populated")
    check("URLs and doc_id values are unique", raw.url.is_unique and raw.doc_id.is_unique, f"rows={len(raw)}")
    check("Preprocessed corpus retains required fields and token output", required.issubset(processed.columns) and "preprocessed_text" in processed.columns, f"rows={len(processed)}")
    check("Inverted index contains postings, DF, and document lengths", bool(index.get("postings")) and bool(index.get("df")) and bool(index.get("doc_lengths")), f"terms={len(index.get('postings', {}))}")
    check("Posting IDs resolve to preprocessed documents", all(p["doc_id"] in set(processed.doc_id) for postings in index["postings"].values() for p in postings), "all posting references resolve")
    check("At least 10 evaluation queries", queries.query_id.nunique() >= 10, f"queries={queries.query_id.nunique()}")
    check("Ground truth has labels and evidence", {"query_id", "doc_id", "label", "evidence"}.issubset(truth.columns) and len(truth) > 0, f"labels={len(truth)}")
    check("Both retrieval models report @10 metrics", set(metrics.model) == {"tfidf", "bm25"} and {"precision@10", "recall@10", "f1@10"}.issubset(metrics.columns), f"rows={len(metrics)}")
    check("Ranking comparison includes both Top-5 lists", {"tfidf_top5", "bm25_top5", "top10_overlap"}.issubset(comparison.columns), f"queries={len(comparison)}")
    check("Search UI, CLI, tests, README, and report exist", all((ROOT / p).exists() for p in ["app.py", "streamlit_app.py", "tests/test_ir.py", "README.md", "requirements.txt"]) and report_path.exists())
    check("Report is laid out with at least eight forced pages", forced_breaks >= 7, f"forced_breaks={forced_breaks}; minimum page sections={forced_breaks + 1}")

    for label, passed, details in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {label}" + (f" — {details}" if details else ""))
    failed = [label for label, passed, _ in checks if not passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        raise SystemExit("Audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
