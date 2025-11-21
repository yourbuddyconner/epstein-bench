from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Protocol, List


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class HFCorpusConfig:
    # Either pass an already loaded dataset, or dataset loading params
    dataset_name: Optional[str] = None          # e.g. "wikipedia"
    dataset_config_name: Optional[str] = None   # e.g. "20220301.en"
    split: str = "train"

    # Column mapping
    text_column: str = "text"           # column with main text
    id_column: Optional[str] = None      # if None, auto-generate IDs

    # Optional: function to build metadata from a row
    metadata_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    # If the user passes an existing Dataset object, we skip load_dataset
    dataset_obj: Any | None = None  # type: ignore

    # Limit number of documents to load (useful for testing)
    limit: Optional[int] = None


@dataclass
class ChunkingConfig:
    max_tokens: int = 256
    overlap_tokens: int = 32


@dataclass
class EmbeddingConfig:
    embed_fn: Callable[[List[str]], List[List[float]]]
    batch_size: int = 32


@dataclass
class SingleHopConfig:
    max_tasks: Optional[int] = None
    max_tasks_per_anchor: int = 1
    min_answer_length: int = 3
    max_answer_length: int = 64


@dataclass
class MultiHopConfig:
    max_tasks: int = 100
    max_hops: int = 2  # v1 supports 2-hop questions
    min_shared_entities: int = 1
    use_llm_entities: bool = False


@dataclass
class RobustnessConfig:
    num_paraphrases: int = 3
    add_typos: bool = True
    typo_rate: float = 0.05
    num_unanswerable: int = 0


@dataclass
class TaskGeneratorConfig:
    # Corpus source
    hf_corpus: Optional[HFCorpusConfig] = None

    # Preprocess
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: Optional[EmbeddingConfig] = None  # optional if not using embedding-based features

    # Task types
    single_hop: Optional[SingleHopConfig] = field(default_factory=SingleHopConfig)
    multi_hop: Optional[MultiHopConfig] = None
    robustness: Optional[RobustnessConfig] = None

    # Global parallelization settings
    max_workers: int = 5
