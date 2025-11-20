"""
Document selection utilities that use the GPT-powered tagging/NER pipeline to
prioritize interesting Epstein Files documents before question generation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm

from src.tagging import GPTOSSNERTagger, TaggingResult


@dataclass
class TaggedDocumentSummary:
    """Lightweight summary of a tagged document used for selection + caching."""

    doc_id: str
    text_hash: str
    score: float
    num_entities: int
    unique_entity_labels: List[str] = field(default_factory=list)
    document_tags: List[str] = field(default_factory=list)
    email_headers: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text_hash": self.text_hash,
            "score": self.score,
            "num_entities": self.num_entities,
            "unique_entity_labels": self.unique_entity_labels,
            "document_tags": self.document_tags,
            "email_headers": self.email_headers,
            "entities": self.entities,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TaggedDocumentSummary":
        return cls(
            doc_id=payload["doc_id"],
            text_hash=payload["text_hash"],
            score=payload.get("score", 0.0),
            num_entities=payload.get("num_entities", 0),
            unique_entity_labels=list(payload.get("unique_entity_labels", [])),
            document_tags=list(payload.get("document_tags", [])),
            email_headers=list(payload.get("email_headers", [])),
            entities=list(payload.get("entities", [])),
        )


class TaggingCache:
    """Disk-backed cache keyed by (doc_id, text_hash)."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt cache - start fresh but keep the file around for debugging.
                backup = self.cache_path.with_suffix(".corrupt")
                try:
                    os.replace(self.cache_path, backup)
                except OSError:
                    pass
                self._data = {}

    def get(self, doc_id: str, text_hash: str) -> Optional[TaggedDocumentSummary]:
        entry = self._data.get(doc_id)
        if not entry:
            return None
        if entry.get("text_hash") != text_hash:
            return None
        summary_payload = entry.get("summary")
        if not summary_payload:
            return None
        return TaggedDocumentSummary.from_dict(summary_payload)

    def set(self, summary: TaggedDocumentSummary) -> None:
        self._data[summary.doc_id] = {
            "text_hash": summary.text_hash,
            "summary": summary.to_dict(),
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        tmp_path = self.cache_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self._data, f)
        os.replace(tmp_path, self.cache_path)
        self._dirty = False


class InterestingDocumentSelector:
    """Ranks documents using GPTOSSNER tagging output."""

    INTERESTING_TAGS = {
        "flight log",
        "flight manifest",
        "legal transcript",
        "deposition",
        "financial record",
        "email",
        "calendar",
        "intel",
        "contact list",
        "travel",
        "roster",
    }

    def __init__(
        self,
        *,
        cache_dir: str = "./data",
        cache_filename: str = "tagging_cache.json",
        max_chars: int = 4000,
        min_confidence: float = 0.55,
        min_entities: int = 3,
        disable_progress: bool = False,
        instructions: Optional[str] = None,
    ) -> None:
        self.cache = TaggingCache(Path(cache_dir) / cache_filename)
        self.max_chars = max_chars
        self.min_confidence = min_confidence
        self.min_entities = min_entities
        self.disable_progress = disable_progress
        self.instructions = instructions
        self.annotations: Dict[str, Dict[str, Any]] = {}
        self.selection_metadata: Dict[str, Any] = {}
        self._tagger: Optional[GPTOSSNERTagger] = None
        self._cache_hits = 0
        self._cache_misses = 0

    def _ensure_tagger(self) -> GPTOSSNERTagger:
        if self._tagger is None:
            self._tagger = GPTOSSNERTagger()
        return self._tagger

    def _truncate_text(self, text: str) -> str:
        return text[: self.max_chars]

    def _score_summary(self, entities: List[Dict[str, Any]], tags: Sequence[str], has_email: bool) -> float:
        if not entities:
            return 0.0

        high_conf = [ent for ent in entities if ent.get("confidence", 0) >= self.min_confidence]
        if len(high_conf) < self.min_entities:
            return 0.0

        label_counts: Dict[str, int] = {}
        for ent in high_conf:
            label = (ent.get("label") or ent.get("type") or "UNKNOWN").upper()
            label_counts[label] = label_counts.get(label, 0) + 1

        unique_labels = len(label_counts)
        tag_bonus = len(tags) * 0.5
        interesting_bonus = sum(1 for tag in tags if tag.lower() in self.INTERESTING_TAGS) * 1.5
        email_bonus = 2.0 if has_email else 0.0
        entity_richness = len(high_conf) * 0.8
        label_diversity = unique_labels * 1.2

        return round(entity_richness + label_diversity + tag_bonus + interesting_bonus + email_bonus, 3)

    def _tag_document(self, text: str) -> TaggingResult:
        tagger = self._ensure_tagger()
        return tagger.analyze_document(text, instructions=self.instructions)

    def _summarize_document(self, doc: Dict[str, Any]) -> TaggedDocumentSummary:
        cached = self.cache.get(doc["doc_id"], doc["text_hash"])
        if cached:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        text = self._truncate_text(doc.get("text", ""))
        result = self._tag_document(text)

        score = self._score_summary(result.entities, result.document_tags, bool(result.email_headers))
        summary = TaggedDocumentSummary(
            doc_id=doc["doc_id"],
            text_hash=doc["text_hash"],
            score=score,
            num_entities=len(result.entities),
            unique_entity_labels=sorted({(ent.get("label") or ent.get("type") or "UNKNOWN").upper() for ent in result.entities}),
            document_tags=result.document_tags,
            email_headers=result.email_headers,
            entities=result.entities,
        )
        self.cache.set(summary)
        return summary

    def select_documents(
        self,
        documents: Sequence[Dict[str, Any]],
        *,
        top_k: Optional[int] = None,
        min_score: float = 2.0,
        candidate_pool_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the highest scoring documents that meet the threshold."""

        if candidate_pool_size:
            sorted_candidates = sorted(
                documents,
                key=lambda d: d.get("text_length", 0),
                reverse=True,
            )[:candidate_pool_size]
        else:
            sorted_candidates = list(documents)

        scored_docs: List[Tuple[float, Dict[str, Any]]] = []
        iterator = tqdm(sorted_candidates, desc="Scoring documents via NER", disable=self.disable_progress)

        for doc in iterator:
            if not doc.get("text"):
                continue

            summary = self._summarize_document(doc)
            summary_dict = summary.to_dict()
            self.annotations[doc["doc_id"]] = summary_dict

            if summary.score < min_score:
                continue

            scored_docs.append((summary.score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        if top_k:
            scored_docs = scored_docs[:top_k]

        selected_docs = [doc for _, doc in scored_docs]
        avg_score = sum(score for score, _ in scored_docs) / len(scored_docs) if scored_docs else 0.0

        self.selection_metadata = {
            "ner_selection_enabled": True,
            "min_score": min_score,
            "top_k": top_k,
            "candidate_pool_size": candidate_pool_size or len(sorted_candidates),
            "retained_documents": len(selected_docs),
            "total_documents": len(documents),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "avg_selected_score": round(avg_score, 3),
            "min_confidence": self.min_confidence,
            "min_entities": self.min_entities,
            "max_chars": self.max_chars,
        }

        self.cache.flush()
        return selected_docs

