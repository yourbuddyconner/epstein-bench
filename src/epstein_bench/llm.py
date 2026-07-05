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
                    "temperature": self.config.temperature,
                    "seed": self.config.seed,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry any transient API error
                last_err = e
                time.sleep(min(2**attempt, 30))
        raise LLMError(f"chat failed after retries: {last_err}")

    # -- embeddings -----------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.config.stub_llm:
            return [_stub_embedding(t) for t in texts]
        client = self._get_client()
        out: list[list[float]] = []
        for start in range(0, len(texts), 256):
            batch = [t[:8000] or " " for t in texts[start : start + 256]]
            last_err: Exception | None = None
            for attempt in range(self.config.max_llm_retries):
                try:
                    resp = client.embeddings.create(
                        model=self.config.embed_model, input=batch
                    )
                    out.extend(d.embedding for d in resp.data)
                    last_err = None
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(min(2**attempt, 30))
            if last_err is not None:
                raise LLMError(f"embed failed after retries: {last_err}")
        return out

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
                    }
                ]
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
    if tag == "BASELINE":
        return json.dumps({"answer": "January 10, 2015", "citations": []})
    return json.dumps({})
