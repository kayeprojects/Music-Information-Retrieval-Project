from collections import Counter
import unittest

from src.preprocessing import normalize_text, preprocess
from src.indexing import build_inverted_index
from src.boolean_retrieval import boolean_search
from src.tfidf_retrieval import cosine_similarity, idf_weight
from src.bm25 import BM25Retriever


def tiny_docs():
    return [
        {"doc_id": "D1", "tokens": ["musik", "musik", "lagu"]},
        {"doc_id": "D2", "tokens": ["musik", "konser"]},
        {"doc_id": "D3", "tokens": ["lagu", "konser", "konser"]},
    ]


class RetrievalTests(unittest.TestCase):
 def test_normalization_and_stemming(self):
    self.assertEqual(normalize_text("Musik—INDONESIA 123"), "musik indonesia")
    self.assertIn("musik", preprocess("Musik Indonesia"))

 def test_postings_and_boolean(self):
    docs = tiny_docs()
    index, _, meta = build_inverted_index(docs)
    self.assertEqual(meta["df"]["musik"], 2)
    self.assertEqual(boolean_search("musik AND lagu", index, {"D1", "D2", "D3"}), {"D1"})
    self.assertEqual(boolean_search("musik OR konser", index, {"D1", "D2", "D3"}), {"D1", "D2", "D3"})

 def test_cosine_and_idf(self):
    self.assertAlmostEqual(cosine_similarity({"x": 1.0}, {"x": 2.0}), 1.0)
    self.assertGreater(idf_weight(10, 1), idf_weight(10, 10))

 def test_bm25_scores_matching_document(self):
    docs = [{"doc_id": "D1", "tokens": ["lagu", "musik"]}, {"doc_id": "D2", "tokens": ["politik"]}]
    index, _, meta = build_inverted_index(docs)
    bm = BM25Retriever(docs, index, meta["df"], lambda q: q.lower().split())
    self.assertGreater(bm.score_document(["lagu"], "D1"), bm.score_document(["lagu"], "D2"))


if __name__ == "__main__":
    unittest.main()
