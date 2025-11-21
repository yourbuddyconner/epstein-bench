from typing import Optional
import openai

from .config import LLMClient


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 512,
        system_prompt: str = "You are a helpful assistant designed to generate evaluation tasks.",
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def generate(self, prompt: str, *, temperature: Optional[float] = None, max_tokens: Optional[int] = None, system_prompt: Optional[str] = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return ""

