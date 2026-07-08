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
from .io_utils import parallel_map, write_jsonl
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


# Prominent figures across politics, tech, finance, entertainment, sport, and
# science, rotated as fabricated false-premise counterparties. Chosen for low
# collision with this corpus; the absence + support checks drop any that a given
# target's documents actually connect to.
_FALSE_PREMISE_FIGURES = (
    "Barack Obama", "Michelle Obama", "Angela Merkel", "Emmanuel Macron",
    "Justin Trudeau", "Narendra Modi", "Pope Francis", "Volodymyr Zelensky",
    "Jacinda Ardern", "Oprah Winfrey", "Taylor Swift", "Beyoncé", "Tom Hanks",
    "Meryl Streep", "LeBron James", "Serena Williams", "Cristiano Ronaldo",
    "Lionel Messi", "Roger Federer", "Greta Thunberg", "Malala Yousafzai",
    "Tim Cook", "Sundar Pichai", "Warren Buffett",
)

# occasions rotated per target so fabricated interactions don't all become
# "a dinner" (a template tell); paired with the diverse figure pool above
_FALSE_PREMISE_OCCASIONS = (
    "private dinner", "one-on-one meeting", "phone call", "conference panel",
    "private flight", "business deal", "charity gala", "working lunch",
    "recorded interview", "video call",
)


# -- per-type generators ---------------------------------------------------------


def gen_single_hop(config: Config, llm: LLM, docs: list[dict], n: int) -> list[dict]:
    rng = random.Random(config.seed)
    clean = [d for d in docs if d["quality"] == "clean"]
    rng.shuffle(clean)

    def per_doc(doc: dict) -> list[dict]:
        prompt = (
            "[FACTS] From the document below, extract up to 3 atomic, verifiable "
            "facts and, for each, write the question an investigative journalist "
            "would ask whose answer is that fact. Rules: the question must name "
            "concrete people/organizations (never 'the document', 'this email', or "
            "bare initials); the answer must be a short span (a name, date, amount, "
            "or quoted phrase) stated verbatim-or-near-verbatim in the document; "
            "skip boilerplate (disclaimers, signatures, mastheads). Also rate each "
            "fact's newsworthiness 1-5: 5 = notable people interacting, unusual "
            "money flows, travel/meetings, legal exposure — the facts a journalist "
            "would report; 1 = administrative trivia (ticket numbers, account "
            "boilerplate, routine scheduling of non-notable staff). The rating "
            "judges importance, never speculation — the fact must still be stated "
            "by the document. Respond with JSON "
            '{"facts": [{"fact": str, "question": str, "answer": str, "salience": int}]}.'
            "\n\n---\n" + _excerpt(doc["text"])
        )
        try:
            resp = llm.chat_json(prompt)
        except LLMError:
            return []
        return [
            _mk_task(
                config,
                "single_hop",
                [doc["doc_id"]],
                question=f["question"].strip(),
                answer=str(f["answer"]).strip(),
                provenance={
                    "generator_model": config.cheap_model,
                    "pipeline_version": __version__,
                    "salience": int(f.get("salience") or 0),
                },
            )
            for f in resp.get("facts", [])
            if f.get("question")
            and f.get("answer")
            and int(f.get("salience") or 0) >= config.min_salience
        ]

    # ~1-3 facts per doc; n docs is comfortably enough for n tasks
    results = parallel_map(per_doc, clean[:n], config.max_workers)
    return [t for batch in results for t in batch][:n]


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

    def per_entity(pair: tuple[str, dict]) -> dict | None:
        name, entity = pair
        edocs = _entity_doc_texts(entity, docs_by_id, max_docs=8)
        if len(edocs) < 2:
            return None
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
            return None
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
            return None
        return _mk_task(
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

    results = parallel_map(per_entity, bounded[: n * 2], config.max_workers)
    return [t for t in results if t][:n]


def gen_timeline(
    config: Config, llm: LLM, docs: list[dict], entities: dict, n: int
) -> list[dict]:
    rng = random.Random(config.seed + 2)
    docs_by_id = {d["doc_id"]: d for d in docs}
    multi = [(name, e) for name, e in entities.items() if len(e["doc_ids"]) >= 2]
    rng.shuffle(multi)

    def per_entity(pair: tuple[str, dict]) -> dict | None:
        _name, entity = pair
        edocs = _entity_doc_texts(entity, docs_by_id, max_docs=4)
        if len(edocs) < 2:
            return None
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
            return None
        if not (resp.get("question") and resp.get("answer")):
            return None
        return _mk_task(
            config,
            "timeline",
            [d["doc_id"] for d in edocs],
            question=str(resp["question"]).strip(),
            answer=str(resp["answer"]).strip(),
        )

    results = parallel_map(per_entity, multi[: n * 2], config.max_workers)
    return [t for t in results if t][:n]


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
    docs_by_id = {d["doc_id"]: d for d in docs}

    def per_doc(doc: dict) -> dict | None:
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
                return None
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
            return None
        if check.get("answerable") is not False:
            return None
        return _mk_task(config, "unanswerable", [], question=question)

    results = parallel_map(per_doc, clean[: n * 2], config.max_workers)
    return [t for t in results if t][:n]


def gen_false_premise(
    config: Config,
    llm: LLM,
    docs: list[dict],
    targets: dict,
    bm25: BM25Retriever,
    n: int,
) -> list[dict]:
    """Questions that presuppose a fabricated fact about a real, in-corpus person.

    Anchored only on entity-complete targets: the corpus holds *every* document
    mentioning them, so 'no document supports this premise' is a bounded,
    checkable claim (the same property that makes dossiers honest, reused for
    negation). A premise is minted by perturbing exactly one element of a real
    fact, then confirmed absent: BM25 top hits for the presupposing question must
    not answer it. Fail closed (drop) on any doubt; the target behavior is
    refusal, so `false_element` records what a system must reject.
    """
    rng = random.Random(config.seed + 4)
    docs_by_id = {d["doc_id"]: d for d in docs}
    # exclude non-person / hub targets (e.g. the "morgan chase" bank fragment)
    # that a false-premise question would render nonsensical
    ordered = [
        (name, info)
        for name, info in targets.items()
        if not any(sub in name.lower() for sub in config.exclude_entity_substrings)
    ]
    rng.shuffle(ordered)
    # a diverse pool of prominent figures across domains, rotated one-per-target
    # so fabricated counterparties don't collapse onto a single name (which would
    # make the type gameable: "mentions Obama -> refuse"). The absence check and
    # support adjudication drop any that a target's documents actually connect to.
    counterparties = list(_FALSE_PREMISE_FIGURES)
    occasions = list(_FALSE_PREMISE_OCCASIONS)
    rng.shuffle(counterparties)
    rng.shuffle(occasions)
    # offset the occasion index from the figure index so the two don't correlate
    assigned = [
        (name, info, counterparties[i % len(counterparties)],
         occasions[(i + 3) % len(occasions)])
        for i, (name, info) in enumerate(ordered)
    ]

    def per_target(quad: tuple[str, dict, str, str]) -> dict | None:
        name, info, figure, occasion = quad
        alias = (info.get("aliases") or [name])[0]
        edocs = [
            docs_by_id[d]
            for d in info["doc_ids"]
            if d in docs_by_id and docs_by_id[d]["quality"] == "clean"
        ][:6]
        if len(edocs) < 2:
            return None
        listing = "\n\n".join(
            f"[DOC {i}]\n{_excerpt(d['text'], 1500)}" for i, d in enumerate(edocs)
        )
        prompt = (
            f'[FALSEPREMISE] The documents below all mention "{alias}". Invent ONE '
            f'question that PRESUPPOSES a FABRICATED {occasion} between EXACTLY TWO '
            f'people — "{alias}" and {figure} — that these documents never mention. '
            f"(If, and only if, {figure} actually appears in the documents below, "
            "substitute a different, comparably prominent public figure from an "
            "unrelated field who is NOT present here.) Name only those two people — no "
            "third party. The entire interaction is invented; pick a plausible year. "
            "Do NOT merely change the date or place of a real meeting — the fabricated "
            f"relationship itself is the false premise. Phrase it naturally as a "
            "question that takes the invented interaction for granted and asks a "
            "follow-up, without signalling that anything is false. Also state the "
            'fabricated interaction in one short phrase. Respond with JSON '
            '{"question": str|null, "false_element": str}.'
            "\n\n" + listing
        )
        try:
            resp = llm.chat_json(prompt)
            question = str(resp.get("question") or "").strip()
            false_element = str(resp.get("false_element") or "").strip()
            if not question or not false_element:
                return None
            top = bm25.search(question, 5)
            absence_doc_ids = [doc_id for doc_id, _ in top if doc_id in docs_by_id]
            context = "\n\n".join(
                f"[DOC id={doc_id}]\n{_excerpt(docs_by_id[doc_id]['text'], 1500)}"
                for doc_id in absence_doc_ids
            )
            check = llm.chat_json(
                "[ABSENT] Can the question below be answered from the documents "
                'provided? Respond with JSON {"answerable": true|false}.'
                f"\n\nQuestion: {question}\n\n{context}"
            )
        except LLMError:
            return None
        if check.get("answerable") is not False:
            return None
        return _mk_task(
            config,
            "false_premise",
            [],
            question=question,
            false_element=false_element,
            provenance={
                "generator_model": config.cheap_model,
                "pipeline_version": __version__,
                "target_entity": name,
                # the most on-topic docs, confirmed here not to support the
                # premise; adjudication re-shows them to confirm fabrication
                "absence_doc_ids": absence_doc_ids,
            },
        )

    results = parallel_map(per_target, assigned[: n * 2], config.max_workers)
    return [t for t in results if t][:n]


def gen_dossier(
    config: Config, llm: LLM, docs: list[dict], targets: dict, n: int
) -> list[dict]:
    """Person-timeline tasks over notable target entities.

    Gold is an item list [(dated event, supporting docs)] — scored like
    aggregation (item-level P/R with citation requirements). Requires the
    entity-complete corpus from scan/select so the timeline is honest.
    """
    rng = random.Random(config.seed + 9)
    docs_by_id = {d["doc_id"]: d for d in docs}
    ordered = list(targets.items())
    rng.shuffle(ordered)

    def per_target(pair: tuple[str, dict]) -> dict | None:
        name, info = pair
        alias = (info.get("aliases") or [name])[0]
        edocs = [
            docs_by_id[d]
            for d in info["doc_ids"]
            if d in docs_by_id and docs_by_id[d]["quality"] == "clean"
        ][:10]
        if len(edocs) < 3:
            return None
        listing = "\n\n".join(
            f"[DOC {i}] id={d['doc_id']}\n{_excerpt(d['text'], 1800)}"
            for i, d in enumerate(edocs)
        )
        prompt = (
            f'[DOSSIER] The documents below all mention "{alias}". Write ONE '
            "timeline question an investigative journalist would ask about this "
            "person's documented activities or interactions (e.g. 'What is the "
            "documented timeline of {alias}'s contact with Jeffrey Epstein?'), "
            "naming the person explicitly. Then answer it as a dated event list: "
            "each item is 'YYYY-MM-DD (or best-available date) — concrete event "
            "stated by a document', with the ids of the documents stating it. "
            "Only events the documents state — no inference of motive or "
            "wrongdoing. Require at least 3 items across at least 2 documents; "
            'if the documents cannot support that, respond {"question": null}. '
            'Respond with JSON {"question": str|null, '
            '"items": [{"item": str, "doc_ids": [str]}]}.'
            "\n\n" + listing
        )
        try:
            resp = llm.chat_json(prompt)
        except LLMError:
            return None
        if not resp.get("question"):
            return None
        valid_ids = {d["doc_id"] for d in edocs}
        items = []
        for it in resp.get("items", []):
            if not it.get("item"):
                continue
            ids = [d for d in (it.get("doc_ids") or []) if d in valid_ids]
            if ids:
                items.append({"item": str(it["item"]).strip(), "doc_ids": ids})
        support = {d for i in items for d in i["doc_ids"]}
        if len(items) < 3 or len(support) < 2:
            return None
        return _mk_task(
            config,
            "dossier",
            sorted(support),
            question=str(resp["question"]).strip(),
            items=items,
            provenance={
                "generator_model": config.cheap_model,
                "pipeline_version": __version__,
                "target_entity": name,
                "candidate_doc_ids": info["doc_ids"],
            },
        )

    results = parallel_map(per_target, ordered[: n * 2], config.max_workers)
    return [t for t in results if t][:n]


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

    # dossier and false_premise both require the entity-complete target set
    if quotas.get("dossier") or quotas.get("false_premise"):
        from .scan import load_targets

        try:
            targets = load_targets(config)
        except FileNotFoundError:
            targets = {}  # direct-corpus mode (no scan/select): skip these types
        if targets:
            if quotas.get("dossier"):
                candidates += gen_dossier(config, llm, docs, targets, quotas["dossier"])
            if quotas.get("false_premise"):
                candidates += gen_false_premise(
                    config, llm, docs, targets, bm25, quotas["false_premise"]
                )

    write_jsonl(config.build_dir / "candidates.jsonl", candidates)
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c["type"]] = counts.get(c["type"], 0) + 1
    counts["total"] = len(candidates)
    return counts
