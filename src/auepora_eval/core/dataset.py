from abc import ABC, abstractmethod
from typing import Iterator, List

from .types import EvaluationSample


class EvaluationDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[EvaluationSample]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...


class InMemoryDataset(EvaluationDataset):
    def __init__(self, name: str, samples: List[EvaluationSample]):
        self._name = name
        self._samples = samples

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self._samples)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

