import uuid
import random
import uuid
import random
import re
from typing import List, Optional

from ..core.types import EvaluationSample, Response, Document
from .config import LLMClient, RobustnessConfig
from .retriever import CorpusRetriever


def build_paraphrase_prompt(query: str) -> str:
    return f"Paraphrase the following question while keeping the meaning exactly the same:\nQuestion: {query}\nParaphrase:"


def add_typos(text: str, rate: float) -> str:
    chars = list(text)
    num_typos = int(len(chars) * rate)
    indices = random.sample(range(len(chars)), num_typos)
    for i in indices:
        # Randomly swap with neighbor or replace
        if i < len(chars) - 1 and random.random() > 0.5:
            chars[i], chars[i+1] = chars[i+1], chars[i]
        else:
            chars[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
    return "".join(chars)


from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

def _process_paraphrase(
    base: EvaluationSample,
    llm: LLMClient,
    cfg: RobustnessConfig
) -> List[EvaluationSample]:
    variants = []
        for _ in range(cfg.num_paraphrases):
            prompt = build_paraphrase_prompt(base.query)
            paraphrased = llm.generate(prompt).strip()
            noisy = add_typos(paraphrased, cfg.typo_rate) if cfg.add_typos else paraphrased

            s = EvaluationSample(
                sample_id=str(uuid.uuid4()),
                query=noisy,
                relevant_docs=base.relevant_docs,
                candidate_docs=base.candidate_docs,
                reference_answer=base.reference_answer,
                labels={
                    **base.labels,
                    "scenario": "paraphrase",
                    "variant_of": base.sample_id,
                },
                metadata={**base.metadata},
            )
            variants.append(s)
    return variants


def generate_paraphrase_variants(
    base_samples: List[EvaluationSample],
    llm: LLMClient,
    cfg: RobustnessConfig,
    max_workers: int = 3
) -> List[EvaluationSample]:
    variants: List[EvaluationSample] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for base in base_samples:
            futures.append(executor.submit(_process_paraphrase, base, llm, cfg))
            
        for future in as_completed(futures):
            result = future.result()
            if result:
                variants.extend(result)

    return variants


NEGATIVE_ANSWER = "There is not enough information in the provided corpus to answer this question."


def _select_sentence_with_entity(text: str, entity: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if entity in sentence:
            return sentence
    return text


NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Drew"]
CITIES = ["Paris", "Berlin", "Sydney", "Toronto", "Madrid", "Lisbon"]


def _mutate_entity(entity: str) -> Optional[str]:
    stripped = entity.strip()
    if not stripped:
        return None

    if stripped.isdigit() and len(stripped) == 4:
        year = int(stripped)
        return str(year + random.randint(5, 25))

    if any(char.isdigit() for char in stripped):
        mutated = re.sub(r"\d", lambda _: str(random.randint(0, 9)), stripped)
        if mutated != stripped:
            return mutated

    parts = stripped.split()
    if len(parts) >= 2:
        return f"{random.choice(NAMES)} {parts[-1]}"

    if stripped in CITIES:
        return random.choice([city for city in CITIES if city != stripped])

    if len(stripped) > 3:
        idx = random.randrange(len(stripped))
        letters = list(stripped)
        letters[idx] = random.choice("abcdefghijklmnopqrstuvwxyz").upper()
        return "".join(letters)

    return None


def _build_unanswerable_prompt(sentence: str, mutated_fact: str) -> str:
    return (
        "You are creating an unanswerable question.\n"
        "Use the provided sentence as inspiration, but make sure the question references the mutated fact which is NOT supported by the sentence.\n"
        "Return only the question.\n\n"
        f"Sentence: {sentence}\n"
        f"Mutated fact: {mutated_fact}\n\n"
        "Question:"
    )


def _attempt_generate_unanswerable(
    chunks: List[Document],
    corpus_texts: List[str],
    llm: LLMClient,
    retriever: Optional[CorpusRetriever]
) -> Optional[EvaluationSample]:
    chunk = random.choice(chunks)
    entities = chunk.metadata.get("entities") or []
    if not entities:
        return None

    entity = random.choice(entities)
    mutated = _mutate_entity(entity)
    if not mutated:
        return None

    mut_lower = mutated.lower()
    # This check can be slow for large corpora, but it's necessary to ensure validity
    if any(mut_lower in text for text in corpus_texts):
        return None

    sentence = _select_sentence_with_entity(chunk.text, entity)
    prompt = _build_unanswerable_prompt(sentence, mutated)
    question = llm.generate(prompt).strip()
    if len(question.split()) < 5:
        return None

    # Optionally fetch misleading docs via retriever
    relevant_docs = [chunk]
    if retriever is not None:
        try:
            distractors = retriever.search(question, k=2)
            relevant_docs = distractors or relevant_docs
        except Exception:
            pass

    return EvaluationSample(
        sample_id=str(uuid.uuid4()),
        query=question,
        relevant_docs=relevant_docs,
        reference_answer=Response(text=NEGATIVE_ANSWER),
        labels={
            "scenario": "unanswerable",
            "source_entity": entity,
        },
    )


def generate_unanswerable_tasks(
    chunks: List[Document],
    llm: LLMClient,
    cfg: RobustnessConfig,
    retriever: Optional[CorpusRetriever] = None,
    max_workers: int = 3
) -> List[EvaluationSample]:
    if cfg.num_unanswerable <= 0:
        return []

    samples: List[EvaluationSample] = []
    attempts = 0
    max_attempts = cfg.num_unanswerable * 10
    corpus_texts = [doc.text.lower() for doc in chunks]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        
        # Initial fill
        while len(futures) < max_workers and attempts < max_attempts:
            futures.add(executor.submit(_attempt_generate_unanswerable, chunks, corpus_texts, llm, retriever))
            attempts += 1
            
        while futures and len(samples) < cfg.num_unanswerable:
            # wait for at least one
            done, not_done = wait(futures, return_when=FIRST_COMPLETED)
            
            futures = not_done
            
            for future in done:
                try:
                    result = future.result()
                    if result:
                        samples.append(result)
                except Exception as e:
                    # Log or ignore exceptions from worker
                    pass
                
                # Refill if needed
                if len(samples) < cfg.num_unanswerable and attempts < max_attempts:
                     futures.add(executor.submit(_attempt_generate_unanswerable, chunks, corpus_texts, llm, retriever))
                     attempts += 1

        # Cancel remaining if any
        for f in futures:
            f.cancel()

    return samples

