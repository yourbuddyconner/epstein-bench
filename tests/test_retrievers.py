from epstein_bench.retrievers import BM25Retriever, HybridRetriever


CHUNKS = [
    {"chunk_id": "a#0", "doc_id": "a", "text": "flight logs list several passengers"},
    {"chunk_id": "b#0", "doc_id": "b", "text": "the painting sold for millions"},
    {"chunk_id": "b#1", "doc_id": "b", "text": "an expensive painting and a buyer"},
    {"chunk_id": "c#0", "doc_id": "c", "text": "committee hearing schedule for june"},
]


def test_bm25_ranks_matching_doc_first():
    r = BM25Retriever(CHUNKS)
    ranked = r.search("painting sold millions", k=3)
    assert ranked[0][0] == "b"


def test_bm25_returns_doc_level_results():
    r = BM25Retriever(CHUNKS)
    ranked = r.search("painting", k=10)
    doc_ids = [d for d, _ in ranked]
    assert doc_ids.count("b") == 1  # two chunks, one doc entry


def test_bm25_no_match_returns_empty():
    r = BM25Retriever(CHUNKS)
    assert r.search("zzzz qqqq", k=5) == []


def test_hybrid_fuses_rankings():
    class Fixed:
        def __init__(self, ranking):
            self.ranking = ranking

        def search(self, query, k):
            return self.ranking[:k]

    a = Fixed([("x", 1.0), ("y", 0.5)])
    b = Fixed([("y", 1.0), ("z", 0.5)])
    fused = HybridRetriever([a, b]).search("q", k=3)
    assert [d for d, _ in fused][0] == "y"  # appears in both lists
