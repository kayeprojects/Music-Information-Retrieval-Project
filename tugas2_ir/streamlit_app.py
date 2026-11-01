"""Streamlit UI entry point for the search engine."""
from pathlib import Path
import streamlit as st

from src.search_engine import SearchEngine

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset_crawling_ir.csv"

st.set_page_config(page_title="ANTARA Musik Indonesia Search", page_icon="🎵", layout="wide")
st.title("Mesin Pencari Artikel Musik Indonesia")
st.caption("Corpus ANTARA News · TF-IDF + cosine atau BM25")

@st.cache_resource
def load_engine():
    return SearchEngine(DATA)

engine = load_engine()
query = st.text_input("Masukkan query", placeholder="contoh: royalti lagu Indonesia")
model_col, k_col = st.columns(2)
with model_col:
    model = st.selectbox("Model ranking", ["bm25", "tfidf"], format_func=lambda x: "BM25" if x == "bm25" else "TF-IDF + cosine")
with k_col:
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
