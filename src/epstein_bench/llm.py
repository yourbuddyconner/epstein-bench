"""LLM access: retried, disk-cached, JSON-first, with a deterministic stub.

Every prompt carries a leading ``[TAG]`` naming the pipeline stage. The stub
implementation dispatches on that tag so the entire pipeline can run end to
end in tests with no API key and fully deterministic output.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import Config


class LLMError(RuntimeError):
    """Raised after retries are exhausted; callers must fail closed."""


class _EmbedSizeError(RuntimeError):
    """Internal: an embeddings request exceeded a token limit (bisect + retry)."""


def _is_size_error(e: Exception) -> bool:
    """True for the embeddings token-limit 400 (per-request or per-item).

    Matched by message so it works regardless of the openai SDK version's
    exception classes. Both observed forms are covered:
      - "maximum request size is 300000 tokens per request"
      - {'code': 'max_tokens_per_request', ...}
    """
    m = str(e).lower()
    return (
        "max_tokens_per_request" in m
        or "maximum request size" in m
        or ("token" in m and "per request" in m)
        or ("maximum context length" in m)  # per-item overflow
    )


def _omit_sampling_params(model: str) -> bool:
    """True for model families that reject ``temperature``/``seed`` on
    chat.completions (they only accept the default temperature=1 and no seed).

    Covers the GPT-5 family and the o-series reasoning models. Verified against
    the API: gpt-5.5 returns 400 "temperature does not support 0 with this
    model". For these, determinism comes from snapshot pinning + the disk cache,
    not from seed.
    """
    m = model.lower()
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))


def _cache_key(model: str, prompt: str, system: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(system.encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


class LLM:
    """Chat + embeddings with on-disk caching keyed by (model, system, prompt)."""

    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self.calls = 0  # cache misses that hit the API (or stub)

    # -- chat ---------------------------------------------------------------

    def chat_json(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str = "You are a precise assistant. Respond with JSON only.",
    ) -> Any:
        """Run a chat completion and parse the response as JSON."""
        text = self.chat(prompt, model=model, system=system, json_mode=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON response: {text[:200]!r}") from e

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str = "You are a precise assistant.",
        json_mode: bool = False,
    ) -> str:
        model = model or self.config.cheap_model
        key = _cache_key(model, prompt, system)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        self.calls += 1
        if self.config.stub_llm:
            text = stub_response(prompt)
        else:
            text = self._openai_chat(model, prompt, system, json_mode)
        self._cache_put(key, text)
        return text

    def _openai_chat(self, model: str, prompt: str, system: str, json_mode: bool) -> str:
        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(self.config.max_llm_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                }
                if not _omit_sampling_params(model):
                    kwargs["temperature"] = self.config.temperature
                    kwargs["seed"] = self.config.seed
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry any transient API error
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise LLMError(f"chat failed after retries: {last_err}")

    # -- embeddings -----------------------------------------------------------
    #
    # The embeddings endpoint enforces a per-request token ceiling AND a
    # per-item token ceiling. OCR text tokenizes unpredictably (BPE explodes on
    # gibberish), so we do NOT try to predict token counts. Instead we pack
    # batches by a conservative char budget and treat the API as the source of
    # truth: any size-limit 400 triggers a recursive bisect (split the batch;
    # for a lone oversized item, halve its characters). This cannot crash on a
    # token limit for any input.

    _EMBED_MAX_CHARS = 8000  # per-item hard cap before the first attempt
    _EMBED_BATCH_CHARS = 250_000  # pack target: safe even at ~1 token/char

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.config.stub_llm:
            return [_stub_embedding(t) for t in texts]
        self._get_client()  # fail fast if no key
        prepared = [t[: self._EMBED_MAX_CHARS] or " " for t in texts]
        out: list[list[float]] = []
        batch: list[str] = []
        chars = 0
        for t in prepared:
            if batch and chars + len(t) > self._EMBED_BATCH_CHARS:
                out.extend(self._embed_adaptive(batch))
                batch, chars = [], 0
            batch.append(t)
            chars += len(t)
        if batch:
            out.extend(self._embed_adaptive(batch))
        return out

    def _embed_adaptive(self, batch: list[str]) -> list[list[float]]:
        """Embed a batch; on a size-limit 400, bisect and retry recursively."""
        try:
            return self._embed_call(batch)
        except _EmbedSizeError:
            if len(batch) == 1:
                # a single item is over the per-item token limit: halve chars.
                # A short string cannot trigger a size error, so this terminates.
                half = max(1, len(batch[0]) // 2)
                return self._embed_adaptive([batch[0][:half]])
            mid = len(batch) // 2
            return self._embed_adaptive(batch[:mid]) + self._embed_adaptive(batch[mid:])

    def _embed_call(self, batch: list[str]) -> list[list[float]]:
        """One embeddings request. Retries transient errors; raises
        _EmbedSizeError on a token-limit 400 so the caller can bisect."""
        client = self._get_client()
        last_err: Exception | None = None
        for attempt in range(self.config.max_llm_retries):
            try:
                resp = client.embeddings.create(
                    model=self.config.embed_model, input=batch
                )
                return [d.embedding for d in resp.data]
            except Exception as e:  # noqa: BLE001
                if _is_size_error(e):
                    raise _EmbedSizeError(str(e)) from e
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise LLMError(f"embed failed after retries: {last_err}")

    # -- plumbing -------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import openai

            if not self.config.openai_api_key:
                raise LLMError("OPENAI_API_KEY is not set and stub mode is off")
            self._client = openai.OpenAI(api_key=self.config.openai_api_key)
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> str | None:
        p = self._cache_path(key)
        try:
            return json.loads(p.read_text())["text"]
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, KeyError):
            # concurrent writer or corrupt entry: treat as a miss and rewrite
            return None

    def _cache_put(self, key: str, text: str) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # atomic replace so a parallel reader never sees a partial file
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"text": text}))
        os.replace(tmp, p)


# -- deterministic stub ---------------------------------------------------------
#
# The stub answers by prompt tag. Tests may monkeypatch `STUB_OVERRIDES` to force
# specific behavior (e.g. a failing verification stage).

STUB_OVERRIDES: dict[str, Any] = {}


def _stub_embedding(text: str, dim: int = 32) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [(h[i % len(h)] - 128) / 128.0 for i in range(dim)]


def stub_response(prompt: str) -> str:
    tag = prompt.split("]", 1)[0].lstrip("[") if prompt.startswith("[") else ""
    if tag in STUB_OVERRIDES:
        override = STUB_OVERRIDES[tag]
        return override(prompt) if callable(override) else override

    if tag == "READABILITY":
        return json.dumps({"readable": True})
    if tag == "FACTS":
        return json.dumps(
            {
                "facts": [
                    {
                        "fact": "Alice Example emailed Bob Sample on 2015-01-10.",
                        "question": "On what date did Alice Example email Bob Sample?",
                        "answer": "January 10, 2015",
                        "salience": 4,
                    }
                ]
            }
        )
    if tag == "NOTABLE":
        return json.dumps({"kind": "person", "notable": True})
    if tag == "DOSSIER":
        return json.dumps(
            {
                "question": "What is the documented timeline of Alice Example's meetings with Bob Sample?",
                "items": [
                    {"item": "2015-01-10 — emailed Bob Sample about the house meeting", "doc_ids": []},
                    {"item": "2015-01-11 — emailed Bob Sample again", "doc_ids": []},
                    {"item": "2015-01-12 — scheduled a committee meeting", "doc_ids": []},
                ],
            }
        )
    if tag == "AGGREGATION":
        return json.dumps(
            {
                "question": "Which people are named in correspondence with Alice Example?",
                "items": [{"item": "Bob Sample", "doc_ids": []}],
            }
        )
    if tag == "TIMELINE":
        return json.dumps(
            {
                "question": "Over what period did Alice Example correspond with Bob Sample?",
                "answer": "January 2015 to March 2015",
            }
        )
    if tag == "UNANSWERABLE":
        return json.dumps(
            {"question": "What was Alice Example's role at Acme Corporation in 1999?"}
        )
    if tag == "FALSEPREMISE":
        return json.dumps(
            {
                "question": "When Alice Example met Bob Sample in Geneva in 2015, "
                "who introduced them?",
                "false_element": "a meeting between Alice Example and Bob Sample in Geneva",
            }
        )
    if tag == "PREMISEID":
        return json.dumps({"identified": True})
    if tag == "STANDALONE":
        return json.dumps({"standalone": True, "reason": "names concrete entities"})
    if tag == "ANSWER":
        return json.dumps({"answer": "January 10, 2015", "found": True})
    if tag in ("CLOSEDBOOK", "DISTRACTOR", "SINGLEDOC"):
        return json.dumps({"answer": None, "found": False})
    if tag == "MATCH":
        return json.dumps({"match": True})
    if tag == "ADJUDICATE":
        return json.dumps({"pass": True, "category": "ok"})
    if tag in ("POOLJUDGE", "POOLRESCUE"):
        return json.dumps({"verdicts": ["supports"] * 16})
    if tag == "SCOREJUDGE":
        return json.dumps({"correct": True, "is_refusal": False})
    if tag == "ABSENT":
        return json.dumps({"answerable": False})
    if tag == "AGGJUDGE":
        return json.dumps({"matched_items": [True], "extra_items": 0})
    if tag == "RECOVER":
        return json.dumps({"present": [True] * 16})
    if tag == "BASELINE":
        return json.dumps({"answer": "January 10, 2015", "citations": []})
    if tag == "PARAMETRIC":
        return json.dumps({"answer": "January 10, 2015", "citations": []})
    return json.dumps({})
