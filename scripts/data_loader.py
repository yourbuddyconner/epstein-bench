"""
Data loader for Epstein Files dataset from Hugging Face.
Downloads and prepares the dataset for NIH benchmark generation.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List
import pickle

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


class EpsteinDataLoader:
    """Handles loading and processing of the Epstein Files dataset."""

    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = None
        self.documents: List[Dict] = []
        self.metadata: Dict[str, any] = {}

    def load_dataset(self, force_reload: bool = False) -> None:
        """Load the Epstein Files dataset from Hugging Face, with on-disk caching."""
        cache_file = self.cache_dir / "epstein_files.pkl"

        if not force_reload and cache_file.exists():
            print("Loading dataset from cache...")
            with open(cache_file, "rb") as f:
                self.dataset = pickle.load(f)
            print(f"Loaded {len(self.dataset)} documents from cache.")
            return

        print("Downloading Epstein Files dataset from Hugging Face...")
        self.dataset = load_dataset("tensonaut/EPSTEIN_FILES_20K", split="train")

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(self.dataset, f)
        print(f"Downloaded and cached {len(self.dataset)} documents.")

    def process_documents(self) -> List[Dict]:
        """Process raw dataset into structured documents with metadata."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        print("Processing documents...")
        self.documents = []

        for idx, item in enumerate(tqdm(self.dataset)):
            raw_text = item.get("text")
            text = "" if raw_text is None else str(raw_text)

            doc = {
                "doc_id": f"epstein_{idx:06d}",
                "filename": item.get("filename", f"unknown_{idx}"),
                "text": text,
                "text_length": len(text),
                "text_hash": hashlib.md5(text.encode()).hexdigest(),
                "source_type": self._determine_source_type(item.get("filename", "")),
                "has_ocr_artifacts": self._detect_ocr_artifacts(text),
            }
            self.documents.append(doc)

        self._generate_metadata()
        return self.documents

    def _determine_source_type(self, filename: str) -> str:
        """Determine if document is from TEXT or IMAGES directory based on filename."""
        if not filename:
            return "UNKNOWN"
        if "IMAGES" in filename or "images" in filename:
            return "OCR"
        if "TEXT" in filename or "text" in filename:
            return "NATIVE"
        return "UNKNOWN"

    def _detect_ocr_artifacts(self, text: str) -> bool:
        """Detect common OCR artifacts in text."""
        if text is None:
            text = ""
        elif not isinstance(text, str):
            text = str(text)

        ocr_indicators = [
            "|",
            "\\",
            "~",
            "  " * 5,  # Multiple spaces
            "\n\n\n",  # Multiple newlines
        ]

        artifact_count = sum(1 for indicator in ocr_indicators if indicator in text[:1000])
        return artifact_count >= 3

    def _generate_metadata(self) -> None:
        """Generate metadata statistics about the dataset."""
        if not self.documents:
            return

        df = pd.DataFrame(self.documents)
        metadata = {
            "total_documents": len(self.documents),
            "total_characters": df["text_length"].sum(),
            "avg_doc_length": df["text_length"].mean(),
            "min_doc_length": df["text_length"].min(),
            "max_doc_length": df["text_length"].max(),
            "source_types": df["source_type"].value_counts().to_dict(),
            "ocr_artifact_count": df["has_ocr_artifacts"].sum(),
            "unique_files": df["filename"].nunique(),
        }

        self.metadata = self._make_json_safe(metadata)

        print("\n=== Dataset Metadata ===")
        for key, value in self.metadata.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            elif isinstance(value, int):
                print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")

    def save_processed_data(self) -> None:
        """Save processed documents and metadata to disk."""
        docs_file = self.cache_dir / "processed_documents.json"
        metadata_file = self.cache_dir / "metadata.json"

        with open(docs_file, "w") as f:
            json.dump(self.documents, f, indent=2)

        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\nSaved processed documents to {docs_file}")
        print(f"Saved metadata to {metadata_file}")

    def search_documents(self, query: str, max_results: int = 10) -> List[Dict]:
        """Simple search function for exploring the dataset."""
        if not self.documents:
            raise ValueError("No documents loaded. Process documents first.")

        query_lower = query.lower()
        results = []

        for doc in self.documents:
            if query_lower in doc["text"].lower():
                text_lower = doc["text"].lower()
                pos = text_lower.index(query_lower)
                start = max(0, pos - 200)
                end = min(len(doc["text"]), pos + len(query) + 200)
                context = doc["text"][start:end]

                results.append(
                    {
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "context": context,
                        "position": pos,
                    }
                )

                if len(results) >= max_results:
                    break

        return results

    def _make_json_safe(self, value):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        if isinstance(value, dict):
            return {k: self._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._make_json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value


if __name__ == "__main__":
    loader = EpsteinDataLoader()
    loader.load_dataset()
    loader.process_documents()
    loader.save_processed_data()

