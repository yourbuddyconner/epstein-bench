from conftest import fixture_rows

from epstein_bench.corpus import (
    build_corpus,
    build_entity_index,
    chunk_text,
    dictionary_ratio,
    garbage_ratio,
    load_chunks,
    load_docs,
    load_entities,
)


def test_garbage_ratio_flags_non_printable():
    assert garbage_ratio("plain ascii text") == 0.0
    assert garbage_ratio("ÿþ□ÿþ□ÿþ□ÿþ□") > 0.5


def test_dictionary_ratio_uses_wordlist():
    wl = frozenset({"the", "meeting", "house"})
    assert dictionary_ratio("the meeting at the house", wl) > 0.5
    assert dictionary_ratio("zx qq vv kk", wl) == 0.0


def test_chunk_text_short_doc_stays_whole():
    assert chunk_text("one two three", chunk_tokens=512, overlap=50) == ["one two three"]


def test_chunk_text_long_doc_windows_overlap():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_tokens=512, overlap=50)
    assert len(chunks) >= 3
    # consecutive chunks share overlap tokens
    assert chunks[0].split()[-1] in chunks[1].split()


def test_entity_index_requires_min_count():
    docs = [
        {"doc_id": f"d{i}", "quality": "clean", "text": "Alice Example wrote again."}
        for i in range(3)
    ] + [{"doc_id": "d9", "quality": "clean", "text": "Carol Once appears here."}]
    index = build_entity_index(docs, min_count=3)
    assert "alice example" in index
    assert "carol once" not in index
    assert index["alice example"]["doc_ids"] == ["d0", "d1", "d2"]


def test_build_corpus_screens_and_chunks(config, llm):
    stats = build_corpus(config, llm, rows=fixture_rows())
    assert stats["docs"] == 10  # the no-text image row is skipped
    assert stats["clean"] == 8
    assert stats["garbage"] >= 1  # the too-short doc
    docs = load_docs(config)
    by_id = {d["doc_id"]: d for d in docs}
    assert "IMG-001" not in by_id
    assert by_id["SHORT-001"]["quality"] == "garbage"
    assert by_id["CLEAN-000"]["quality"] == "clean"
    assert by_id["CLEAN-000"]["meta"]["file_type"] == "pdf"
    # garbage docs never enter the retrieval corpus
    chunk_docs = {c["doc_id"] for c in load_chunks(config)}
    assert "SHORT-001" not in chunk_docs
    # entity alias index picked up recurring names
    entities = load_entities(config)
    assert "alice example" in entities
    assert len(entities["alice example"]["doc_ids"]) == 8
