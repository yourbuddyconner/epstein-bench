from abc import ABC, abstractmethod
from typing import Protocol, List

from .types import EvaluationSample, SystemOutputs, RetrievedDocument, Document, Response


class RAGSystem(ABC):
    @abstractmethod
    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        """Run the full RAG pipeline on a given sample.

        Responsibilities:
        - Perform retrieval using `sample.query`.
        - Perform generation using the retrieved docs (and possibly `sample` fields).
        - Return retrieved docs and final response wrapped in `SystemOutputs`.
        """
        raise NotImplementedError


# Optional protocol type for duck-typing
class RAGSystemProtocol(Protocol):
    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs: ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedDocument]:
        ...


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, context_docs: List[Document]) -> Response:
        ...


class SimpleRAGSystem(RAGSystem):
    def __init__(self, retriever: Retriever, generator: Generator):
        self._retriever = retriever
        self._generator = generator

    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        retrieved = self._retriever.retrieve(sample.query, top_k=top_k)
        docs = [r.doc for r in retrieved]
        response = self._generator.generate(sample.query, docs)
        return SystemOutputs(retrieved=retrieved, response=response)

