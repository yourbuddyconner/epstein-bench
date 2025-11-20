"""NER and document tagging powered by OpenAI GPT models or local GPT-OSS."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]

from .config import AppConfig, load_config

LOGGER = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "DATE",
    "MONEY",
    "EMAIL",
    "PHONE",
    "EVENT",
    "DOCUMENT",
    "TAIL_NUMBER",
]

SYSTEM_PROMPT = (
    "You are a meticulous intelligence analyst. Extract all relevant named "
    "entities, apply high-level document tags, and capture any structured "
    "email headers (from/to/cc/bcc). Respond ONLY with JSON "
    "following this schema: {\"entities\": [{\"label\": str, \"text\": str, "
    "\"confidence\": float, \"evidence\": str}], \"document_tags\": [str], "
    "\"email_headers\": [{\"from\": str, \"to\": [str], \"cc\": [str], "
    "\"bcc\": [str], \"subject\": str, \"sent_at\": str, \"confidence\": float, "
    "\"evidence\": str}]}. Confidence must be a float between 0 and 1. "
    "Evidence should quote or paraphrase supporting text snippets."
)


@dataclass
class TaggingResult:
    """Structured output returned from the tagging pipelines."""

    entities: List[Dict[str, Any]] = field(default_factory=list)
    document_tags: List[str] = field(default_factory=list)
    email_headers: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


class GPTOSSNERTagger:
    """Helper that can run tagging via OpenAI API or local GPT-OSS inference."""

    SUPPORTED_BACKENDS = ("openai", "local")

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        entity_types: Optional[Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
        enable_email_headers: bool = True,
        backend: Optional[str] = None,
    ) -> None:
        """Initialize the tagger with configurable defaults."""

        self._config: AppConfig = load_config()
        requested_backend = (backend or self._config.tagging_backend).lower()
        if requested_backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported tagging backend '{requested_backend}'. "
                f"Valid options: {', '.join(self.SUPPORTED_BACKENDS)}"
            )

        self.backend = requested_backend
        self.model_name = model_name or self._config.tagging_model_name
        self.entity_types = list(entity_types or DEFAULT_ENTITY_TYPES)
        self.max_new_tokens = max_new_tokens or self._config.tagging_max_output_tokens
        self.temperature = (
            temperature if temperature is not None else self._config.tagging_temperature
        )
        self.enable_email_headers = enable_email_headers
        self._client: Any = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForCausalLM] = None

        if self.backend == "local":
            preferred_device = (
                device
                or self._config.tagging_device
                or self._config.embeddings_device
                or self._config.default_device
            )
            self.device = self._config.resolve_device(preferred_device)
        else:
            self.device = device  # Unused for OpenAI backend.

    def _build_messages(self, text: str, instructions: Optional[str]) -> List[Dict[str, str]]:
        """Construct chat messages for the tagging prompt."""

        entity_schema = ", ".join(self.entity_types)
        email_note = (
            " If the passage resembles an email (headers, quoted metadata, etc.), "
            "capture structured email_headers entries with from/to/cc/bcc/subject/sent_at."
            if self.enable_email_headers
            else ""
        )
        custom_instructions = instructions or ""
        system_prompt = SYSTEM_PROMPT + f" Target entity labels: [{entity_schema}]." + email_note
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{custom_instructions}\n\nDocument:\n{text}"},
        ]

    def _ensure_client(self) -> Any:
        """Return a cached OpenAI client, initializing it on first use."""

        if self.backend != "openai":
            raise RuntimeError("OpenAI client requested while backend is not 'openai'.")

        if self._client is not None:
            return self._client

        if OpenAI is None:
            raise ImportError(
                "openai is required for GPTOSSNERTagger when using the 'openai' backend. "
                "Install it with `pip install openai>=1.0.0`."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable must be set to use the OpenAI tagging backend."
            )

        self._client = OpenAI(api_key=api_key)
        return self._client

    def _ensure_local_model(self) -> None:
        """Load the local GPT-OSS model into memory if needed."""

        if self.backend != "local":
            raise RuntimeError("Local model requested while backend is not 'local'.")

        if self._tokenizer is not None and self._model is not None:
            return

        token_kwargs = {}
        if self._config.huggingface_token:
            token_kwargs["token"] = self._config.huggingface_token

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "transformers is required for the 'local' tagging backend. "
                "Install it with `pip install transformers`."
            )
        if torch is None:
            raise ImportError(
                "torch is required for the 'local' tagging backend. Install it with `pip install torch`."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **token_kwargs)

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16
        elif self.device.startswith("mps") and torch.backends.mps.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **token_kwargs,
        )
        self._model.to(self.device)
        self._model.eval()

    def _generate_openai_completion(self, text: str, instructions: Optional[str]) -> str:
        """Call the OpenAI Chat Completions API and return the raw content."""

        client = self._ensure_client()
        messages = self._build_messages(text, instructions)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:  # pragma: no cover - network errors
            LOGGER.exception("OpenAI tagging request failed.")
            raise RuntimeError("OpenAI tagging request failed") from exc

        message = response.choices[0].message
        content = message.content or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(content, str):
            raise RuntimeError("OpenAI response did not contain text content.")
        return content.strip()

    def _generate_local_completion(self, text: str, instructions: Optional[str]) -> str:
        """Run local GPT-OSS generation and return the decoded response."""

        self._ensure_local_model()
        assert self._tokenizer is not None
        assert self._model is not None

        messages = self._build_messages(text, instructions)
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        completion_tokens = output_ids[:, input_ids.shape[-1] :]
        completion = self._tokenizer.decode(completion_tokens[0], skip_special_tokens=True).strip()
        return completion

    def _generate_completion(self, text: str, instructions: Optional[str]) -> str:
        """Dispatch to the appropriate backend for completion generation."""

        if self.backend == "openai":
            return self._generate_openai_completion(text, instructions)
        if self.backend == "local":
            return self._generate_local_completion(text, instructions)
        raise RuntimeError(f"Unsupported backend '{self.backend}'")

    def analyze_document(self, text: str, *, instructions: Optional[str] = None) -> TaggingResult:
        """Run tagging on arbitrary text via the configured backend.

        If the text is long, it is split into chunks to avoid context window or output token limits.
        Results are then merged.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Simple character-based heuristic for chunking.
        # 15,000 chars is roughly 3-4k tokens. This leaves plenty of room for system prompts
        # and generated JSON within standard context windows (e.g. 8k or 128k),
        # but helps keep the *generated* JSON size manageable per chunk.
        CHUNK_SIZE = 15000
        if len(text) <= CHUNK_SIZE:
            completion = self._generate_completion(text, instructions)
            return self._parse_completion(completion)

        # Split text into chunks (preserve word boundaries if possible)
        chunks = self._split_text(text, CHUNK_SIZE)
        results: List[TaggingResult] = []

        LOGGER.info("Document too large (%d chars). Split into %d chunks.", len(text), len(chunks))

        # Determine concurrency
        max_workers = 1
        if self.backend == "openai":
            max_workers = self._config.tagging_max_concurrent_requests

        if max_workers > 1:
            LOGGER.info("Processing chunks concurrently with %d workers.", max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._generate_completion, chunk, instructions): i
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete, but we need to order them? 
                # Actually, _merge_results doesn't strictly rely on order for entities, 
                # but order is nice for raw_response log. 
                # Let's just collect them and sort by index if needed, or just append.
                # For now, appending in completion order is fine or we can sort.
                # To keep it deterministic, let's store by index.
                indexed_results: List[tuple[int, TaggingResult]] = []
                
                for future in as_completed(future_to_chunk):
                    idx = future_to_chunk[future]
                    try:
                        completion = future.result()
                        indexed_results.append((idx, self._parse_completion(completion)))
                    except Exception as exc:
                        LOGGER.error(f"Chunk {idx} failed: {exc}")
                        # We might want to return a partial result or re-raise.
                        # For now, treat as empty result to avoid crashing whole doc?
                        # Or maybe just let it fail. The original code didn't handle loop errors explicitly.
                        # Let's just log and continue with partial.
                        indexed_results.append((idx, TaggingResult(raw_response=f"Error processing chunk {idx}: {exc}")))
                
                # Sort by index to preserve document order in raw_response
                indexed_results.sort(key=lambda x: x[0])
                results = [r for _, r in indexed_results]

        else:
            for i, chunk in enumerate(chunks):
                LOGGER.debug("Processing chunk %d/%d...", i + 1, len(chunks))
                completion = self._generate_completion(chunk, instructions)
                results.append(self._parse_completion(completion))

        return self._merge_results(results)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters."""
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to find a newline or space to break on
            last_newline = text.rfind("\n", start, end)
            if last_newline != -1 and last_newline > start + chunk_size // 2:
                split_at = last_newline + 1
            else:
                last_space = text.rfind(" ", start, end)
                if last_space != -1 and last_space > start + chunk_size // 2:
                    split_at = last_space + 1
                else:
                    # Hard break if no good boundary found
                    split_at = end

            chunks.append(text[start:split_at])
            start = split_at

        return chunks

    def _merge_results(self, results: List[TaggingResult]) -> TaggingResult:
        """Combine multiple TaggingResult objects into one."""
        if not results:
            return TaggingResult()

        combined_entities: List[Dict[str, Any]] = []
        combined_tags: List[str] = []
        combined_headers: List[Dict[str, Any]] = []
        raw_parts: List[str] = []

        for r in results:
            combined_entities.extend(r.entities)
            combined_tags.extend(r.document_tags)
            combined_headers.extend(r.email_headers)
            if r.raw_response:
                raw_parts.append(r.raw_response)

        # Deduplicate document tags (preserve order)
        seen_tags = set()
        unique_tags = []
        for t in combined_tags:
            if t not in seen_tags:
                seen_tags.add(t)
                unique_tags.append(t)

        return TaggingResult(
            entities=combined_entities,
            document_tags=unique_tags,
            email_headers=combined_headers,
            raw_response="\n\n--- CHUNK SEPARATOR ---\n\n".join(raw_parts),
        )

    def extract_entities(self, text: str, *, instructions: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convenience method returning only entities."""

        return self.analyze_document(text, instructions=instructions).entities

    def extract_email_headers(self, text: str, *, instructions: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return structured email metadata (from/to/cc/bcc/etc.) when present."""

        return self.analyze_document(text, instructions=instructions).email_headers

    def tag_document(self, text: str, *, instructions: Optional[str] = None) -> List[str]:
        """Convenience method returning only document-level tags."""

        return self.analyze_document(text, instructions=instructions).document_tags

    def _parse_completion(self, completion: str) -> TaggingResult:
        """Parse the JSON payload returned from either backend."""

        json_payload = self._extract_json(completion)
        if json_payload is None:
            snippet = completion.strip()
            # if len(snippet) > 300:
            #     snippet = snippet[:297] + "..."
            LOGGER.warning(
                "Unable to parse tagging response; returning raw text. Snippet: %s",
                snippet,
            )
            return TaggingResult(raw_response=completion)

        entities = json_payload.get("entities", [])
        tags = json_payload.get("document_tags", [])
        email_headers = json_payload.get("email_headers", [])
        return TaggingResult(
            entities=entities,
            document_tags=tags,
            email_headers=email_headers,
            raw_response=completion,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None

        candidates: List[str] = []

        # Prefer fenced code blocks that declare JSON
        code_block_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
        candidates.extend(match.group(1) for match in code_block_pattern.finditer(text))

        # Fallback to the broadest brace-enclosed snippet
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            candidates.append(text[brace_start : brace_end + 1])

        for candidate in candidates:
            cleaned = GPTOSSNERTagger._sanitize_json_candidate(candidate)
            payload = GPTOSSNERTagger._loads_json_loose(cleaned)
            if payload is not None:
                return payload

        return None

    @staticmethod
    def _sanitize_json_candidate(candidate: str) -> str:
        """Normalize smart quotes, code-fence preambles, and stray commas."""
        cleaned = candidate.strip()
        if cleaned.lower().startswith("json"):
            brace_index = cleaned.find("{")
            if brace_index != -1:
                cleaned = cleaned[brace_index:]

        replacements = {
            "\u201c": '"',
            "\u201d": '"',
            "\u2018": "'",
            "\u2019": "'",
            "\u00a0": " ",
            "\ufeff": "",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        return GPTOSSNERTagger._escape_newlines_inside_strings(cleaned)

    @staticmethod
    def _escape_newlines_inside_strings(payload: str) -> str:
        """Replace literal newline characters inside JSON strings with \\n."""
        result_chars: List[str] = []
        in_string = False
        escape = False

        for ch in payload:
            if in_string:
                if escape:
                    result_chars.append(ch)
                    escape = False
                    continue
                if ch == "\\":
                    result_chars.append(ch)
                    escape = True
                    continue
                if ch == '"':
                    result_chars.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    result_chars.append("\\n")
                    continue
                if ch == "\r":
                    # Skip bare carriage returns to avoid introducing \r sequences
                    continue
                result_chars.append(ch)
                continue
            else:
                if ch == '"':
                    in_string = True
                result_chars.append(ch)

        return "".join(result_chars)

    @staticmethod
    def _loads_json_loose(candidate: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse slightly malformed JSON payloads."""
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Remove trailing commas that JSON doesn't tolerate
        no_trailing_commas = re.sub(r",(\s*[}\]])", r"\1", candidate)
        if no_trailing_commas != candidate:
            try:
                return json.loads(no_trailing_commas)
            except json.JSONDecodeError:
                pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed  # type: ignore[return-value]
        except (ValueError, SyntaxError):
            return None

        return None

