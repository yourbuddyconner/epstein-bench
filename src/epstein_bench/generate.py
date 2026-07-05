"""Fact-first candidate task generation.

Candidates are drafted from *clean* documents only, then oversampled ~3x the
final target; the verification gauntlet (verify.py) culls them. Every
candidate carries provenance: source doc ids, generator model, pipeline
version.

Candidate schema (build/candidates.jsonl):
    {task_id, type, question, answer, items, source_doc_ids, provenance}
- answer: short string (single_hop, timeline); null for others
- items: [{"item": str, "doc_ids": [str]}] for aggregation; null for others
"""

from __future__ import annotations

import random
import uuid

from . import __version__
from .config import Config
from .corpus import load_docs, load_entities
from .io_utils import write_jsonl
from .llm import LLM, LLMError
from .retrievers import BM25Retriever


def _mk_task(config: Config, type_: str, source_doc_ids: list[str], **fields) -> dict:
    return {
        "task_id": str(uuid.uuid4()),
        "type": type_,
        "answer": None,
        "items": None,
        "source_doc_ids": source_doc_ids,
        "provenance": {
            "generator_model": config.cheap_model,
            "pipeline_version": __version__,
        },
        **fields,
    }


def _excerpt(text: str, limit: int = 3000) -> str:
    return text[:limit]


# -- per-type generators ---------------------------------------------------------


def gen_single_hop(config: Config, llm: LLM, docs: list[dict], n: int) -> list[dict]:
    rng = random.Random(config.seed)
    clean = [d for d in docs if d["quality"] == "clean"]
    rng.shuffle(clean)
    out: list[dict] = []
    for doc in clean:
        if len(out) >= n:
            break
        prompt = (
            "[FACTS] From the document below, extract up to 3 atomic, verifiable "
            "facts and, for each, write the question an investigative journalist "
            "would ask whose answer is that fact. Rules: the question must name "
            "concrete people/organizations (never 'the document', 'this email', or "
            "bare initials); the answer must be a short span (a name, date, amount, "
            "or quoted phrase) stated verbatim-or-near-verbatim in the document; "
            "skip boilerplate (disclaimers, signatures, mastheads). Respond with "
            'JSON {"facts": [{"fact": str, "question": str, "answer": str}]}.'
            "\n\n---\n" + _excerpt(doc["text"])
        )
        try:
            resp = llm.chat_json(prompt)
        except LLMError:
            continue
        for f in resp.get("facts", []):
            if not (f.get("question") and f.get("answer")):
                continue
            out.append(
                _mk_task(
                    config,
                    "single_hop",
                    [doc["doc_id"]],
                    question=f["question"].strip(),
                    answer=str(f["answer"]).strip(),
                )
            )
    return out[:n]


def _entity_doc_texts(
    entity: dict, docs_by_id: dict[str, dict], max_docs: int
) -> list[dict]:
    return [
        docs_by_id[d]
        for d in entity["doc_ids"][:max_docs]
        if d in docs_by_id and docs_by_id[d]["quality"] == "clean"
    ]


def gen_aggregation(
    config: Config, llm: LLM, docs: list[dict], entities: dict, n: int
) -> list[dict]:
    rng = random.Random(config.seed + 1)
    docs_by_id = {d["doc_id"]: d for d in docs}
    # bounded scope: entities whose full candidate doc set is enumerable
    bounded = [
        (name, e) for name, e in entities.items() if 2 <= len(e["doc_ids"]) <= 15
    ]
    rng.shuffle(bounded)
    out: list[dict] = []
    for name, entity in bounded:
        if len(out) >= n:
            break
        edocs = _entity_doc_texts(entity, docs_by_id, max_docs=8)
        if len(edocs) < 2:
            continue
        listing = "\n\n".join(
            f"[DOC {i}] id={d['doc_id']}\n{_excerpt(d['text'], 1500)}"
            for i, d in enumerate(edocs)
        )
        prompt = (
            f"[AGGREGATION] The documents below all mention \"{entity['aliases'][0]}\". "
            "Write ONE list-style question an investigator would ask that is answered "
            "by aggregating across at least two of these documents (e.g. 'Which "
            "people are named in correspondence with X about Y?'). The question must "
            "name the entity explicitly. Then answer it: each item must be a short "
            "concrete phrase, with the ids of the documents that support it. Respond "
            'with JSON {"question": str, "items": [{"item": str, "doc_ids": [str]}]}.'
            "\n\n" + listing
        )
        try:
            resp = llm.chat_json(prompt)
        except LLMError:
            continue
        items = [
            {"item": str(i["item"]).strip(), "doc_ids": list(i.get("doc_ids") or [])}
            for i in resp.get("items", [])
            if i.get("item")
        ]
        valid_ids = {d["doc_id"] for d in edocs}
        for item in items:
            item["doc_ids"] = [d for d in item["doc_ids"] if d in valid_ids]
        items = [i for i in items if i["doc_ids"]]
        support = {d for i in items for d in i["doc_ids"]}
        if not resp.get("question") or len(items) < 2 or len(support) < 2:
            continue
        out.append(
            _mk_task(
                config,
                "aggregation",
                sorted(support),
                question=resp["question"].strip(),
                items=items,
                provenance={
                    "generator_model": config.cheap_model,
                    "pipeline_version": __version__,
                    "bounding_entity": name,
                    "candidate_doc_ids": entity["doc_ids"],
                },
            )
        )
    return out[:n]


def gen_timeline(
    config: Config, llm: LLM, docs: list[dict], entities: dict, n: int
) -> list[dict]:
    rng = random.Random(config.seed + 2)
    docs_by_id = {d["doc_id"]: d for d in docs}
    multi = [(name, e) for name, e in entities.items() if len(e["doc_ids"]) >= 2]
    rng.shuffle(multi)
    out: list[dict] = []
    for name, entity in multi:
        if len(out) >= n:
            break
        edocs = _entity_doc_texts(entity, docs_by_id, max_docs=4)
        if len(edocs) < 2:
            continue
        listing = "\n\n".join(
            f"[DOC {i}] id={d['doc_id']}\n{_excerpt(d['text'], 1500)}"
            for i, d in enumerate(edocs)
        )
        prompt = (
            f"[TIMELINE] The documents below all mention \"{entity['aliases'][0]}\". "
            "If (and only if) at least two documents carry distinct dates or clear "
            "temporal ordering, write ONE question about the timing, ordering, or "
            "time span of events that requires at least two of the documents to "
            "answer, naming the entity explicitly, plus its short factual answer. "
            'Respond with JSON {"question": str|null, "answer": str|null}.'
            "\n\n" + listing
        )
        try:
            resp = llm.chat_json(prompt)
        except LLMError:
            continue
        if not (resp.get("question") and resp.get("answer")):
            continue
        out.append(
            _mk_task(
                config,
                "timeline",
                [d["doc_id"] for d in edocs],
                question=str(resp["question"]).strip(),
                answer=str(resp["answer"]).strip(),
            )
        )
    return out[:n]


def gen_unanswerable(
    config: Config, llm: LLM, docs: list[dict], bm25: BM25Retriever, n: int
) -> list[dict]:
    """Plausible questions verified absent from the corpus.

    Absence check: retrieve top BM25 docs for the drafted question and have the
    LLM confirm none of them answer it. Fail closed (drop) on any doubt.
    """
    rng = random.Random(config.seed + 3)
    clean = [d for d in docs if d["quality"] == "clean"]
    rng.shuffle(clean)
    out: list[dict] = []
    docs_by_id = {d["doc_id"]: d for d in docs}
    for doc in clean:
        if len(out) >= n:
            break
        prompt = (
            "[UNANSWERABLE] Read the document below, then invent ONE question in "
            "the same domain that sounds like it could be answered by this document "
            "collection but is about a detail NOT stated here (e.g. change the "
            "role, year, place, or counterparty of a real fact). The question must "
            "name concrete entities and must not be answerable from this document. "
            'Respond with JSON {"question": str}.\n\n---\n' + _excerpt(doc["text"])
        )
        try:
            resp = llm.chat_json(prompt)
            question = str(resp.get("question") or "").strip()
            if not question:
                continue
            top = bm25.search(question, 5)
            context = "\n\n".join(
                f"[DOC id={doc_id}]\n{_excerpt(docs_by_id[doc_id]['text'], 1500)}"
                for doc_id, _ in top
                if doc_id in docs_by_id
            )
            check = llm.chat_json(
                "[ABSENT] Can the question below be answered from the documents "
                'provided? Respond with JSON {"answerable": true|false}.'
                f"\n\nQuestion: {question}\n\n{context}"
            )
        except LLMError:
            continue
        if check.get("answerable") is not False:
            continue
        out.append(_mk_task(config, "unanswerable", [], question=question))
    return out[:n]


# -- stage entry point --------------------------------------------------------------


def generate_candidates(config: Config, llm: LLM) -> dict[str, int]:
    docs = load_docs(config)
    entities = load_entities(config)
    from .corpus import load_chunks

    bm25 = BM25Retriever(load_chunks(config))

    quotas = {
        t: int(config.target_tasks * share * config.oversample_factor)
        for t, share in config.type_mix.items()
    }
    candidates: list[dict] = []
    candidates += gen_single_hop(config, llm, docs, quotas["single_hop"])
    candidates += gen_aggregation(config, llm, docs, entities, quotas["aggregation"])
    candidates += gen_timeline(config, llm, docs, entities, quotas["timeline"])
    candidates += gen_unanswerable(config, llm, docs, bm25, quotas["unanswerable"])

    write_jsonl(config.build_dir / "candidates.jsonl", candidates)
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    counts["total"] = len(candidates)
    return counts
