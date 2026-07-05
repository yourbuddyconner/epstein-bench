"""Reference baselines: bm25, dense, hybrid, and closed_book (no retrieval).

Each reads a released ``questions.jsonl`` and emits a spec-conformant
``predictions.jsonl`` — exactly the file contract any outside system uses.
The closed-book baseline exists as public evidence that the tasks require
retrieval.

Usage:
    python baselines/run_baseline.py --system hybrid --split dev --out preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epstein_bench import DATASET_VERSION  # noqa: E402
from epstein_bench.config import Config  # noqa: E402
from epstein_bench.corpus import load_chunks, load_docs  # noqa: E402
from epstein_bench.io_utils import read_jsonl, write_jsonl  # noqa: E402
from epstein_bench.llm import LLM, LLMError  # noqa: E402
from epstein_bench.retrievers import build_retrievers  # noqa: E402

REFUSAL = "The corpus does not contain enough information to answer this question."
TOP_K = 20  # retrieved list length reported (contract max)
CONTEXT_DOCS = 5  # docs shown to the generator


def answer_with_context(
    llm: LLM, question: str, context_docs: list[dict]
) -> tuple[str, list[str]]:
    listing = "\n\n".join(
        f"[DOC id={d['doc_id']}]\n{d['text'][:2000]}" for d in context_docs
    )
    prompt = (
        "[BASELINE] Answer the question using ONLY the documents provided, and "
        "cite the ids of the documents that support your answer. If the documents "
        'do not contain the answer, respond {"answer": null, "citations": []}. '
        'Respond with JSON {"answer": str|null, "citations": [str]}.'
        f"\n\nQuestion: {question}\n\n{listing}"
    )
    resp = llm.chat_json(prompt)
    answer = resp.get("answer")
    citations = [str(c) for c in (resp.get("citations") or [])]
    if not answer:
        return REFUSAL, []
    return str(answer), citations


def answer_closed_book(llm: LLM, question: str) -> str:
    prompt = (
        "[BASELINE] Answer this question from your own knowledge. If you do not "
        'know, respond {"answer": null, "citations": []}. Respond with JSON '
        '{"answer": str|null, "citations": []}.'
        f"\n\nQuestion: {question}"
    )
    resp = llm.chat_json(prompt)
    return str(resp.get("answer")) if resp.get("answer") else REFUSAL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", required=True, choices=["bm25", "dense", "hybrid", "closed_book"]
    )
    parser.add_argument("--split", default="dev", choices=["dev", "full"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = Config()
    llm = LLM(config)
    questions = list(
        read_jsonl(
            config.dataset_dir / DATASET_VERSION / args.split / "questions.jsonl"
        )
    )

    retriever = None
    docs_by_id: dict[str, dict] = {}
    if args.system != "closed_book":
        chunks = load_chunks(config)
        docs_by_id = {d["doc_id"]: d for d in load_docs(config)}
        retriever = build_retrievers(config, chunks, llm)[args.system]

    predictions = []
    for q in questions:
        retrieved: list[str] = []
        try:
            if retriever is None:
                answer, citations = answer_closed_book(llm, q["question"]), []
            else:
                ranked = retriever.search(q["question"], TOP_K)
                retrieved = [doc_id for doc_id, _ in ranked]
                context = [
                    docs_by_id[d] for d in retrieved[:CONTEXT_DOCS] if d in docs_by_id
                ]
                answer, citations = answer_with_context(llm, q["question"], context)
        except LLMError as e:
            print(f"warn: {q['task_id']} failed ({e}); recording refusal", file=sys.stderr)
            answer, citations = REFUSAL, []
        predictions.append(
            {
                "task_id": q["task_id"],
                "answer": answer,
                "citations": citations,
                "retrieved": retrieved,
            }
        )

    n = write_jsonl(args.out, predictions)
    print(json.dumps({"system": args.system, "split": args.split, "predictions": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
