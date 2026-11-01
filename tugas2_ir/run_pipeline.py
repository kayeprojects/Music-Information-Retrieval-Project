"""Prepare task-2 artifacts from the Task-1 CSV located in the project root."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.search_engine import SearchEngine
from src.preprocessing import preprocess, stemmer_name

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
SOURCE_CSV = PROJECT_ROOT / "dataset_musik_indonesia_antaranews.csv"
DATA_DIR = ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus_csv = DATA_DIR / "dataset_crawling_ir.csv"
    if not corpus_csv.exists():
        if not SOURCE_CSV.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {SOURCE_CSV}")
        shutil.copy2(SOURCE_CSV, corpus_csv)
    engine = SearchEngine(corpus_csv)
    engine.save_artifacts(DATA_DIR / "dataset_preprocessed.csv", ROOT / "index" / "inverted_index.json")
    sample = engine.raw_df.iloc[0]
    print(f"Loaded {len(engine.documents)} documents; vocabulary={len(engine.index)}; stemmer={stemmer_name()}")
    print("Before:", sample.title, str(sample.content)[:240])
    print("After:", " ".join(preprocess(f"{sample.title} {str(sample.content)[:240]}"))[:300])
    print("Artifacts written: data/dataset_crawling_ir.csv, data/dataset_preprocessed.csv, index/inverted_index.json")
    print("Sample terms:")
    shown = [term for term in ("musik", "lagu", "album", "royalti", "konser") if term in engine.index]
    for term in shown:
        print(term, "DF=", engine.df[term], "postings=", engine.index[term][:5])
    print("Example query — BM25")
    from src.search_engine import display_results
    display_results(engine.search("royalti lagu Indonesia", model="bm25", top_k=5))


if __name__ == "__main__":
    main()
