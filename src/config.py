import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

@dataclass
class Config:
    # Hugging Face
    huggingface_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN")

    # Embeddings
    embeddings_model_name: str = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    embeddings_device: Optional[str] = os.getenv("EMBEDDINGS_DEVICE")

    # Tagging / LLM Backend
    tagging_backend: str = os.getenv("TAGGING_BACKEND", "openai")
    tagging_model_name: str = os.getenv("TAGGING_MODEL_NAME", "gpt-4o-mini")
    tagging_max_output_tokens: int = int(os.getenv("TAGGING_MAX_OUTPUT_TOKENS", "512"))
    tagging_temperature: float = float(os.getenv("TAGGING_TEMPERATURE", "0.2"))
    tagging_device: Optional[str] = os.getenv("TAGGING_DEVICE")

    # OpenAI
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Default Device
    default_device: Optional[str] = os.getenv("DEFAULT_DEVICE")

    def __post_init__(self):
        # Basic validation or fallback logic could go here
        pass

# Global config instance
config = Config()

