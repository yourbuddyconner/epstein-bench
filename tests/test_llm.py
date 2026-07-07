"""LLM client internals: embedding batching must respect the token budget."""

from epstein_bench.llm import LLM


class _FakeEmbeddings:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, model, input):
        # record each batch's estimated token size; return dummy vectors
        self.recorder.append(sum(len(t) // 4 + 1 for t in input))

        class _R:
            data = [type("D", (), {"embedding": [0.0, 1.0]})() for _ in input]

        return _R()


class _FakeClient:
    def __init__(self, recorder):
        self.embeddings = _FakeEmbeddings(recorder)


def test_embed_batches_within_token_budget(config, monkeypatch):
    config.stub_llm = False  # exercise the real batching path
    llm = LLM(config)
    batches: list[int] = []
    llm._client = _FakeClient(batches)

    # 400 chunks of ~2000 tokens each = ~800k tokens; must split into batches
    texts = ["word " * 1600 for _ in range(400)]  # ~8000 chars -> ~2000 tokens
    vecs = llm.embed(texts)

    assert len(vecs) == 400  # every input embedded, order preserved
    assert len(batches) > 1  # actually split
    assert all(b <= LLM._EMBED_TOKEN_BUDGET for b in batches)  # never over budget


def test_embed_truncates_long_texts(config):
    config.stub_llm = False
    llm = LLM(config)
    batches: list[int] = []
    llm._client = _FakeClient(batches)
    llm.embed(["x" * 100_000])  # single pathological chunk
    assert batches[0] <= LLM._EMBED_MAX_CHARS // 4 + 1  # truncated before send
