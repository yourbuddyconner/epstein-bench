import uuid
import random
import logging
from typing import List, Dict, Tuple, Optional, Set

from ..core.types import Document, EvaluationSample, Response
from .anchors import Anchor
from .config import LLMClient, MultiHopConfig

logger = logging.getLogger(__name__)

def _lookup_chunk(chunks: List[Document], doc_id: str) -> Document:
    for c in chunks:
        if c.doc_id == doc_id:
            return c
    raise ValueError(f"Chunk {doc_id} not found")


def _llm_entities(llm: LLMClient, text: str) -> Set[str]:
    prompt = (
        "Extract up to five named entities (people, organisations, locations) from the text.\n"
        "Return them as a comma separated list with no additional text.\n\n"
        f"Text:\n{text}\n\nEntities:"
    )
    response = llm.generate(prompt).strip()
    entities = [part.strip() for part in response.split(",") if part.strip()]
    return set(entities)


def extract_entities(text: str, *, use_llm: bool = False, llm: Optional[LLMClient] = None) -> Set[str]:
    """
    Extract entities from text using heuristic (capitalised phrases) or an LLM.
    """
    if use_llm and llm is not None:
        try:
            entities = _llm_entities(llm, text)
            if entities:
                return entities
        except Exception as exc:
            logger.warning("LLM entity extraction failed; falling back to heuristic. Error: %s", exc)

    entities: Set[str] = set()
    words = text.split()
    if not words:
        return entities

    i = 0
    while i < len(words):
        word = words[i].strip(".,!?;:()[]\"'")
        if word and word[0].isupper() and len(word) > 2:
            entity_parts = [word]
            j = i + 1
            while j < len(words):
                next_word = words[j].strip(".,!?;:()[]\"'")
                if next_word and next_word[0].isupper():
                    entity_parts.append(next_word)
                    j += 1
                else:
                    break

            entity = " ".join(entity_parts)
            if entity.lower() not in {"the", "a", "an", "this", "that", "these", "those", "there", "here", "what", "when", "where", "who", "why", "how"}:
                entities.add(entity)

            i = j
        else:
            i += 1

    return entities


def build_entity_graph(chunks: List[Document], cfg: MultiHopConfig, llm: Optional[LLMClient] = None) -> Dict[str, List[str]]:
    """
    Build a graph where nodes are doc_ids and edges exist if they share entities.
    """
    inv_index: Dict[str, List[str]] = {}
    graph: Dict[str, List[str]] = {}

    logger.info("Building entity graph from %d chunks...", len(chunks))

    for i, chunk in enumerate(chunks):
        graph[chunk.doc_id] = []

        entities = extract_entities(
            chunk.text,
            use_llm=cfg.use_llm_entities,
            llm=llm,
        )
        chunk.metadata["entities"] = list(entities)

        for entity in entities:
            inv_index.setdefault(entity, []).append(chunk.doc_id)

        if i % 1000 == 0 and i > 0:
            logger.info("Processed %d chunks for entity graph...", i)

    shared_counts: Dict[Tuple[str, str], int] = {}
    for entity, doc_ids in inv_index.items():
        unique_ids = list(set(doc_ids))
        if len(unique_ids) <= 1:
            continue
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                pair = tuple(sorted((unique_ids[i], unique_ids[j])))
                shared_counts[pair] = shared_counts.get(pair, 0) + 1

    edge_count = 0
    for (id1, id2), count in shared_counts.items():
        if count < cfg.min_shared_entities:
            continue
        if id2 not in graph[id1]:
            graph[id1].append(id2)
            edge_count += 1
        if id1 not in graph[id2]:
            graph[id2].append(id1)
            edge_count += 1

    logger.info("Entity graph built. %d nodes, %d edges.", len(graph), edge_count)
    return graph


def sample_doc_pairs(graph: Dict[str, List[str]], max_pairs: int) -> List[Tuple[str, str]]:
    pairs = []
    nodes = list(graph.keys())
    random.shuffle(nodes)
    
    for u in nodes:
        if not graph[u]:
            continue
        v = random.choice(graph[u])
        
        # Avoid self-loops (though graph building logic usually prevents them if using distinct indices)
        if u == v: 
            continue
            
        pairs.append((u, v))
        if len(pairs) >= max_pairs:
            break
            
    return pairs


def build_multi_hop_prompt(s1: str, s2: str) -> str:
    return (
        f"Fact 1: {s1}\n"
        f"Fact 2: {s2}\n\n"
        "Task: Create a multi-hop question that requires combining information from both facts to answer. "
        "Also provide the answer.\n"
        "Format:\n"
        "Question: <question>\n"
        "Answer: <answer>\n"
    )


def parse_multi_hop_output(text: str) -> Optional[Tuple[str, str]]:
    lines = text.strip().split('\n')
    q = None
    a = None
    
    # More robust parsing
    current_section = None
    q_lines = []
    a_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        lower_line = line.lower()
        if lower_line.startswith("question:"):
            current_section = "q"
            q_lines.append(line[9:].strip())
        elif lower_line.startswith("answer:"):
            current_section = "a"
            a_lines.append(line[7:].strip())
        elif current_section == "q":
            q_lines.append(line)
        elif current_section == "a":
            a_lines.append(line)
            
    if q_lines and a_lines:
        q = " ".join(q_lines).strip()
        a = " ".join(a_lines).strip()
        return q, a
        
    return None


from concurrent.futures import ThreadPoolExecutor, as_completed

def _process_multi_hop_pair(
    doc_id1: str,
    doc_id2: str,
    anchors_by_doc: Dict[str, List[Anchor]],
    chunks: List[Document],
    llm: LLMClient
) -> Optional[EvaluationSample]:
    anchors1 = anchors_by_doc.get(doc_id1, [])
    anchors2 = anchors_by_doc.get(doc_id2, [])
    if not anchors1 or not anchors2:
        return None

    a1 = random.choice(anchors1)
    a2 = random.choice(anchors2)

    prompt = build_multi_hop_prompt(a1.sentence, a2.sentence)
    out = llm.generate(prompt)
    parsed = parse_multi_hop_output(out)
    
    if parsed is None:
        return None

    question, answer_text = parsed

    try:
        relevant_docs = [
            _lookup_chunk(chunks, doc_id1),
            _lookup_chunk(chunks, doc_id2),
        ]
    except ValueError:
        return None

    return EvaluationSample(
            sample_id=str(uuid.uuid4()),
            query=question,
            relevant_docs=relevant_docs,
            reference_answer=Response(text=answer_text),
            labels={"type": "multi_hop", "hops": 2},
        )


def generate_multi_hop_tasks(
    chunks: List[Document],
    anchors_by_doc: Dict[str, List[Anchor]],
    llm: LLMClient,
    cfg: MultiHopConfig,
    max_workers: int = 3
) -> Tuple[List[EvaluationSample], Dict[str, List[str]]]:
    # 1. Build Graph
    graph = build_entity_graph(chunks, cfg, llm if cfg.use_llm_entities else None)
    
    # 2. Sample pairs
    # Request more pairs than needed to account for failures
    pairs = sample_doc_pairs(graph, cfg.max_tasks * 3)

    samples: List[EvaluationSample] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for doc_id1, doc_id2 in pairs:
            futures.append(executor.submit(
                _process_multi_hop_pair, 
                doc_id1, doc_id2, anchors_by_doc, chunks, llm
            ))
            
        for future in as_completed(futures):
            result = future.result()
            if result:
                samples.append(result)
                if len(samples) >= cfg.max_tasks:
                    for f in futures:
                        f.cancel()
                    break

    return samples, graph
