import uuid
from typing import List

from ..core.types import Document, EvaluationSample, Response
from .anchors import Anchor
from .config import LLMClient, SingleHopConfig


def choose_answer_span(anchor: Anchor) -> str:
    # Simple baseline: choose a named entity or noun phrase via heuristic.
    # For v1, just pick the first capitalized phrase that isn't the start of the sentence.
    words = anchor.sentence.split()
    if not words:
        return ""
        
    # Skip first word to avoid starting the sentence
    for i in range(1, len(words)):
        if words[i][0].isupper():
            # Grab contiguous capitalized words
            span = [words[i]]
            for j in range(i + 1, len(words)):
                if words[j][0].isupper():
                    span.append(words[j])
                else:
                    break
            return " ".join(span).strip(".,!?;:")
            
    # Fallback: last word
    return words[-1].strip(".,!?;:")


def find_relevant_docs(answer_span: str, chunks: List[Document]) -> List[Document]:
    # Baseline: simple substring match over chunk.text.
    relevant = []
    for chunk in chunks:
        if answer_span in chunk.text:
            relevant.append(chunk)
    return relevant


def build_single_hop_prompt(sentence: str, answer: str) -> str:
    return (
        f"Context: {sentence}\n"
        f"Answer: {answer}\n\n"
        "Task: Write a question based on the Context such that the Answer is exactly the answer to the question. "
        "The question should be fully answerable using only the information in the Context.\n"
        "Question:"
    )


import uuid
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.types import Document, EvaluationSample, Response
from .anchors import Anchor
from .config import LLMClient, SingleHopConfig


def choose_answer_span(anchor: Anchor) -> str:
    # Simple baseline: choose a named entity or noun phrase via heuristic.
    # For v1, just pick the first capitalized phrase that isn't the start of the sentence.
    words = anchor.sentence.split()
    if not words:
        return ""
        
    # Skip first word to avoid starting the sentence
    for i in range(1, len(words)):
        if words[i][0].isupper():
            # Grab contiguous capitalized words
            span = [words[i]]
            for j in range(i + 1, len(words)):
                if words[j][0].isupper():
                    span.append(words[j])
                else:
                    break
            return " ".join(span).strip(".,!?;:")
            
    # Fallback: last word
    return words[-1].strip(".,!?;:")


def find_relevant_docs(answer_span: str, chunks: List[Document]) -> List[Document]:
    # Baseline: simple substring match over chunk.text.
    relevant = []
    for chunk in chunks:
        if answer_span in chunk.text:
            relevant.append(chunk)
    return relevant


def build_single_hop_prompt(sentence: str, answer: str) -> str:
    return (
        f"Context: {sentence}\n"
        f"Answer: {answer}\n\n"
        "Task: Write a question based on the Context such that the Answer is exactly the answer to the question. "
        "The question should be fully answerable using only the information in the Context.\n"
        "Question:"
    )


def _process_anchor(
    anchor: Anchor,
    chunks: List[Document],
    llm: LLMClient,
    cfg: SingleHopConfig
) -> Optional[EvaluationSample]:
        answer_span = choose_answer_span(anchor)
        if not answer_span:
        return None
            
        # Check length constraints
        token_len = len(answer_span.split())
        if not (cfg.min_answer_length <= token_len <= cfg.max_answer_length):
        return None

        # 1) Generate question
        prompt = build_single_hop_prompt(anchor.sentence, answer_span)
        question = llm.generate(prompt).strip()

        # 2) Find relevant docs
        rel_docs = find_relevant_docs(answer_span, chunks)
        if not rel_docs:
        return None

        # 3) Build reference answer
        ref_answer = Response(text=answer_span)

    return EvaluationSample(
            sample_id=str(uuid.uuid4()),
            query=question,
            relevant_docs=rel_docs,
            reference_answer=ref_answer,
            labels={"type": "single_hop_factoid"},
            metadata={"source_anchor_doc_id": anchor.doc_id},
        )


def generate_single_hop_tasks(
    anchors: List[Anchor],
    chunks: List[Document],
    llm: LLMClient,
    cfg: SingleHopConfig,
    max_workers: int = 3
) -> List[EvaluationSample]:
    samples: List[EvaluationSample] = []
    
    # If we have a limit, we might not need to process all anchors.
    # But since filters apply, we might need to process more than max_tasks.
    # We'll process in batches or just all if list isn't huge. 
    # Anchors list can be large, so let's process them.
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # Submit all anchors? If anchors is huge (e.g. 20k), this creates 20k futures.
        # Maybe just slice if we want to be safe, but anchors are from chunks.
        # For now submit all.
        for anchor in anchors:
            futures.append(executor.submit(_process_anchor, anchor, chunks, llm, cfg))
            
        for future in as_completed(futures):
            result = future.result()
            if result:
                samples.append(result)
                if cfg.max_tasks is not None and len(samples) >= cfg.max_tasks:
                    # Cancel remaining
                    for f in futures:
                        f.cancel()
                    break

    return samples

