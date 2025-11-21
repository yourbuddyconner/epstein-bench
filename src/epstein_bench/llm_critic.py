from __future__ import annotations

import re
from typing import Optional

from ..auepora_eval.metrics.generation_llm_judge import LLMCritic
from ..auepora_eval.taskgen.openai_client import OpenAIClient
from ..config import config


class OpenAIChatCritic(LLMCritic):
    """LLMCritic implementation backed by OpenAI Chat completions."""

    SCORE_RE = re.compile(r"([0-9]*\.?[0-9]+)")

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 64,
        system_prompt: str = "You are a meticulous grader. Only respond with a numeric score between 0 and 1.",
    ):
        self.client = OpenAIClient(
            api_key=api_key or config.openai_api_key,
            model=model or config.tagging_model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

    def score(self, *, prompt: str, metadata: Optional[dict] = None) -> float:
        raw = self.client.generate(prompt, temperature=0.0)
        match = self.SCORE_RE.search(raw)
        if not match:
            return 0.0
        try:
            value = float(match.group(1))
        except ValueError:
            value = 0.0
        return max(0.0, min(1.0, value))

