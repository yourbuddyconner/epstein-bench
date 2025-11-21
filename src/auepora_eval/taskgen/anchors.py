from dataclasses import dataclass
from typing import Dict, Any, List

from ..core.types import Document


@dataclass
class Anchor:
    doc_id: str
    sentence: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


def extract_anchors(chunks: List[Document], max_anchors_per_chunk: int = 3) -> List[Anchor]:
    """
    For each chunk:
      * Split into sentences.
      * Score sentences with simple heuristics:
        * Length within a range.
        * Contains at least one capitalized token or number.
      * Keep top N per chunk.
      * Fill metadata with { "chunk_doc_id": chunk.doc_id } and any extra info.
    """
    anchors: List[Anchor] = []

    for chunk in chunks:
        # Simple sentence splitting by period (naive for v1)
        # In production, use nltk.sent_tokenize or similar
        raw_sentences = chunk.text.split('.')
        candidates = []
        
        current_char_idx = 0
        
        for s in raw_sentences:
            s_stripped = s.strip()
            if not s_stripped:
                current_char_idx += len(s) + 1 # +1 for the split char
                continue
                
            # Heuristic scoring
            length = len(s_stripped.split())
            score = 0.0
            
            # Length range preference
            if 10 <= length <= 50:
                score += 1.0
            
            # Capitalization or numbers
            if any(w[0].isupper() for w in s_stripped.split() if w):
                score += 0.5
            if any(c.isdigit() for c in s_stripped):
                score += 0.5

            if score > 0:
                candidates.append((score, s_stripped, current_char_idx))
            
            current_char_idx += len(s) + 1

        # Sort by score desc
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top N
        top_n = candidates[:max_anchors_per_chunk]
        
        for _, sent, start in top_n:
            anchors.append(Anchor(
                doc_id=chunk.doc_id, # Using chunk id as anchor doc id reference
                sentence=sent,
                start_char=start,
                end_char=start + len(sent),
                metadata={"chunk_doc_id": chunk.doc_id}
            ))

    return anchors

