from dataclasses import dataclass
from typing import List, Optional
import pickle
import os
import logging
from pathlib import Path

from ..core.types import Document
from .config import ChunkingConfig, EmbeddingConfig

logger = logging.getLogger(__name__)

@dataclass
class PreprocessResult:
    chunks: List[Document]
    embeddings: List[List[float]]  # same length as chunks


def chunk_documents(docs: List[Document], cfg: ChunkingConfig) -> List[Document]:
    """
    Split doc.text into token chunks with overlap;
    maintain metadata and add {"parent_doc_id": doc.doc_id, "chunk_index": i}.
    """
    chunks: List[Document] = []
    
    for doc in docs:
        text = doc.text
        # Naive whitespace chunking for v1 (spec doesn't mandate a specific tokenizer)
        # In a real impl, use a tokenizer.
        words = text.split()
        
        if not words:
            continue

        step = cfg.max_tokens - cfg.overlap_tokens
        if step < 1:
            step = 1
            
        for i, start_idx in enumerate(range(0, len(words), step)):
            end_idx = start_idx + cfg.max_tokens
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunk_id = f"{doc.doc_id}_chunk_{i}"
            meta = doc.metadata.copy()
            meta.update({
                "parent_doc_id": doc.doc_id, 
                "chunk_index": i
            })
            
            chunks.append(Document(doc_id=chunk_id, text=chunk_text, metadata=meta))
            
    return chunks


def embed_chunks(chunks: List[Document], cfg: EmbeddingConfig, cache_dir: Optional[str] = ".cache/embeddings") -> List[List[float]]:
    """
    Call cfg.embed_fn in batches.
    Supports caching to disk to avoid re-computing embeddings.
    """
    
    # 1. Check cache if enabled
    cache_file = None
    if cache_dir:
        import hashlib
        # Create a stable hash of the input texts to key the cache
        # This assumes chunks are in stable order
        content_hash = hashlib.md5("".join(d.doc_id for d in chunks).encode("utf-8")).hexdigest()
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"embeddings_{content_hash}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading embeddings from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")

    # 2. Compute embeddings
    all_embeddings: List[List[float]] = []
    texts = [d.text for d in chunks]
    
    for i in range(0, len(texts), cfg.batch_size):
        batch = texts[i : i + cfg.batch_size]
        batch_embeddings = cfg.embed_fn(batch)
        all_embeddings.extend(batch_embeddings)

    # 3. Write cache
    if cache_file:
        try:
            logger.info(f"Saving embeddings to cache: {cache_file}")
            with open(cache_file, "wb") as f:
                pickle.dump(all_embeddings, f)
        except Exception as e:
            logger.warning(f"Failed to write embedding cache: {e}")
        
    return all_embeddings
