from typing import List

from datasets import load_dataset, Dataset, DatasetDict

from ..core.types import Document
from .config import HFCorpusConfig


class HFCorpusAdapter:
    def __init__(self, config: HFCorpusConfig):
        self.config = config
        self._dataset: Dataset | None = None

    def load(self) -> Dataset:
        if self.config.dataset_obj is not None:
            ds = self.config.dataset_obj
            # If DatasetDict, pick split
            if isinstance(ds, DatasetDict):
                self._dataset = ds[self.config.split]
            else:
                self._dataset = ds
        else:
            ds = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config_name,
                split=self.config.split,
            )
            self._dataset = ds
        
        if self.config.limit is not None and self._dataset is not None:
            self._dataset = self._dataset.select(range(min(len(self._dataset), self.config.limit)))
            
        return self._dataset

    def to_documents(self) -> List[Document]:
        if self._dataset is None:
            self.load()
        assert self._dataset is not None

        docs: List[Document] = []
        for idx, row in enumerate(self._dataset):
            doc_id = (
                str(row[self.config.id_column])
                if self.config.id_column is not None
                else f"hf_doc_{idx}"
            )
            text = row[self.config.text_column]
            if text is None:
                text = ""
            metadata = (
                self.config.metadata_fn(row)
                if self.config.metadata_fn is not None
                else {}
            )
            docs.append(Document(doc_id=doc_id, text=text, metadata=metadata))

        return docs
