"""SentenceTransformer-based embedding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import AppConfig, load_config

TextInput = Union[str, Sequence[str]]


@dataclass
class EmbeddingClient:
    """Thin wrapper around SentenceTransformer with sane defaults."""

    model_name: str | None = None
    device: str | None = None
    normalize_embeddings: bool = True
    batch_size: int = 32

    _model: SentenceTransformer | None = None
    _config: AppConfig | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        self._config = load_config()
        model_name = self.model_name or self._config.embeddings_model_name
        target_device = self._config.resolve_device(self.device or self._config.embeddings_device)
        self._model = SentenceTransformer(model_name, device=target_device)
        return self._model

    def embed(self, texts: TextInput) -> np.ndarray:
        """Return embeddings for a string or sequence of strings."""

        model = self._ensure_model()
        single = isinstance(texts, str)
        data: List[str]
        if single:
            data = [texts]
        else:
            data = [str(t) for t in texts]  # type: ignore[arg-type]

        vectors = model.encode(
            data,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return vectors[0] if single else vectors

    def embed_dataframe(
        self,
        frame: pd.DataFrame,
        text_column: str,
        *,
        embedding_column: str = "embedding",
        dropna: bool = True,
        copy_frame: bool = True,
    ) -> pd.DataFrame:
        """Add embeddings to a dataframe and return the updated frame.

        Parameters
        ----------
        frame:
            Source dataframe containing a ``text_column`` with raw strings.
        text_column:
            Column to embed.
        embedding_column:
            Name of the column where vector results will be stored.
        dropna:
            Skip rows where ``text_column`` is null before computing embeddings.
        copy_frame:
            When True (default) operates on a shallow copy. Set to False to write
            embeddings directly onto the supplied dataframe.
        """

        target_df = frame.copy(deep=False) if copy_frame else frame
        valid_mask = (
            target_df[text_column].notna()
            if dropna
            else pd.Series(True, index=target_df.index, dtype=bool)
        )

        if embedding_column not in target_df.columns:
            target_df[embedding_column] = None
        else:
            target_df.loc[:, embedding_column] = None

        if not valid_mask.any():
            return target_df

        texts = target_df.loc[valid_mask, text_column].astype(str).tolist()
        vectors = self.embed(texts)
        vector_series = pd.Series(list(vectors), index=target_df.index[valid_mask], dtype="object")
        target_df.loc[valid_mask, embedding_column] = vector_series
        return target_df

