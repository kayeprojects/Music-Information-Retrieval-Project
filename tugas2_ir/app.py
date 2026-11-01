"""Simple Streamlit search UI; command line mode works without Streamlit."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.search_engine import SearchEngine, display_results  # noqa: E402

DATA = ROOT / "data" / "dataset_crawling_ir.csv"


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="ANTARA Musik Indonesia search engine")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--model", choices=["tfidf", "bm25"], default="bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--boolean", action="store_true", help="Interpret query as Boolean AND/OR/NOT")
    args = parser.parse_args()
    engine = SearchEngine(DATA)
    query = args.query or input("Query: ")
    if args.boolean:
        for row in engine.boolean(query):
            print(f"{row['doc_id']} | {row['title']} | {row['url']}")
    else:
        display_results(engine.search(query, args.model, args.top_k))


def run_streamlit() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit belum terpasang. Install requirements: pip install -r requirements.txt") from exc
    st.set_page_config(page_title="ANTARA Musik Indonesia Search", page_icon="🎵", layout="wide")
    st.title("Mesin Pencari Artikel Musik Indonesia")
    st.caption("Corpus artikel ANTARA News · TF-IDF + cosine atau BM25")
    @st.cache_resource
    def get_engine():
        return SearchEngine(DATA)
    engine = get_engine()
    query = st.text_input("Masukkan query", placeholder="contoh: royalti lagu Indonesia")
    left, right = st.columns(2)
    with left:
        model = st.selectbox("Model ranking", ["bm25", "tfidf"], format_func=lambda x: "BM25" if x == "bm25" else "TF-IDF + cosine")
    with right:
        top_k = st.slider("Top-k", 1, 20, 10)
    if st.button("Cari", type="primary") and query.strip():
        results = engine.search(query, model, top_k)
        if not results:
            st.info("Tidak ditemukan dokumen yang cocok. Coba query atau istilah lain.")
        for result in results:
            st.markdown(f"**{result['rank']}. [{result['title']}]({result['url']})** — skor {result['score']:.4f}")
            st.write(result["snippet"])
    st.divider()
    st.caption(f"Corpus: {len(engine.documents):,} dokumen · {len(engine.index):,} term terindeks")


if __name__ == "__main__":
    if "streamlit" in sys.argv[0].lower():
        run_streamlit()
    else:
        run_cli()
