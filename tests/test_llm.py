"""Embedding batching must never exceed the API's token limit, for any input.

We don't predict token counts (OCR tokenizes unpredictably); we treat the API
as the source of truth and bisect on a size-limit 400. These tests use a fake
client that enforces a hard size limit and rejects oversized requests exactly
like OpenAI does.
"""

import pytest

from epstein_bench.llm import LLM


class _FakeBadRequest(Exception):
    pass


class _FakeEmbeddings:
    """Rejects any request whose total 'tokens' (chars here) exceed limit,
    and any single item over item_limit — mirroring the two OpenAI ceilings."""

    def __init__(self, limit, item_limit, accepted):
        self.limit = limit
        self.item_limit = item_limit
        self.accepted = accepted  # records total size of each ACCEPTED request

    def create(self, model, input):
        total = sum(len(t) for t in input)
        if any(len(t) > self.item_limit for t in input):
            raise _FakeBadRequest("maximum request size is N tokens per request")
        if total > self.limit:
            raise _FakeBadRequest("max_tokens_per_request exceeded")
        self.accepted.append(total)

        class _R:
            data = [type("D", (), {"embedding": [float(len(t)), 1.0]})() for t in input]

        return _R()


class _FakeClient:
    def __init__(self, limit, item_limit, accepted):
        self.embeddings = _FakeEmbeddings(limit, item_limit, accepted)


def _llm_with_fake(config, limit, item_limit):
    config.stub_llm = False
    llm = LLM(config)
    accepted: list[int] = []
    llm._client = _FakeClient(limit, item_limit, accepted)
    return llm, accepted


def test_embed_never_exceeds_limit_and_preserves_order(config):
    llm, accepted = _llm_with_fake(config, limit=1000, item_limit=1000)
    # 50 texts of 300 chars each = 15000 "tokens"; must split into many requests
    texts = ["a" * 300 for _ in range(50)]
    vecs = llm.embed(texts)
    assert len(vecs) == 50  # all embedded
    assert vecs[0][0] == 300.0  # order/content preserved (embedding encodes len)
    assert accepted and all(t <= 1000 for t in accepted)  # never over the ceiling


def test_embed_bisects_a_batch_that_overflows_after_packing(config):
    # pack budget is huge, so up-front packing makes one big batch; the fake
    # rejects it and the adaptive bisect must recover.
    llm, accepted = _llm_with_fake(config, limit=1000, item_limit=1000)
    llm._EMBED_BATCH_CHARS = 10_000_000  # force a single oversized initial batch
    vecs = llm.embed(["a" * 300 for _ in range(20)])
    assert len(vecs) == 20
    assert all(t <= 1000 for t in accepted)


def test_embed_truncates_a_single_oversized_item(config):
    # one item far exceeds the per-item limit; must be halved until it fits.
    llm, accepted = _llm_with_fake(config, limit=100_000, item_limit=500)
    # _EMBED_MAX_CHARS caps at 8000 first, then bisect halves 8000->...-><=500
    vecs = llm.embed(["b" * 100_000])
    assert len(vecs) == 1
    assert accepted and accepted[0] <= 500


def test_embed_stub_mode_bypasses_api(config):
    config.stub_llm = True
    llm = LLM(config)
    vecs = llm.embed(["hello", "world"])
    assert len(vecs) == 2 and len(vecs[0]) == 32
