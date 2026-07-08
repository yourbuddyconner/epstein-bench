"""Reference baselines, built on the optional producer-side SDK.

Each baseline is a ``System`` (see ``epstein_bench.sdk``); this script wires one
up and runs it through ``sdk.run`` to emit a spec-conformant
``predictions.jsonl`` — exactly the file contract any outside system uses.

    python baselines/run_baseline.py --system hybrid --split dev --out preds.jsonl
    python baselines/run_baseline.py --system agentic --model claude-sonnet-5 \
        --split full --out preds.jsonl   # needs ANTHROPIC_API_KEY

Systems: bm25, dense, hybrid (retrieval), closed_book, parametric (no retrieval,
evidence that the tasks require retrieval), and agentic (an LLM tool-use agent
on the Anthropic Messages API — a stronger reference).
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
from epstein_bench.llm import LLM  # noqa: E402
from epstein_bench.retrievers import build_retrievers  # noqa: E402
from epstein_bench.sdk import run  # noqa: E402
from epstein_bench.sdk.agentic import AgenticRAG  # noqa: E402
from epstein_bench.sdk.systems import NoContextSystem, RetrievalSystem  # noqa: E402

RETRIEVAL_SYSTEMS = ("bm25", "dense", "hybrid")


def build_system(config: Config, llm: LLM, system: str, model: str | None):
    if system in ("closed_book", "parametric"):
        return NoContextSystem(llm, system)

    docs_by_id = {d["doc_id"]: d for d in load_docs(config)}
    chunks = load_chunks(config)
    if system in RETRIEVAL_SYSTEMS:
        retriever = build_retrievers(config, chunks, llm)[system]
        return RetrievalSystem(llm, retriever, docs_by_id)

    if system == "agentic":
        import anthropic  # optional dep; only needed for this baseline

        # the agent reasons over retrieved evidence; hybrid RRF is the strongest
        # reference retriever to hand it
        retriever = build_retrievers(config, chunks, llm)["hybrid"]
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return AgenticRAG(client, model or config.agent_model, retriever, docs_by_id)

    raise ValueError(f"unknown system: {system}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        required=True,
        choices=[*RETRIEVAL_SYSTEMS, "closed_book", "parametric", "agentic"],
    )
    parser.add_argument("--split", default="dev", choices=["dev", "full"])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model", help="model id for the agentic system (default config.agent_model)"
    )
    args = parser.parse_args()

    config = Config()
    llm = LLM(config)
    system = build_system(config, llm, args.system, args.model)

    questions_path = config.dataset_dir / DATASET_VERSION / args.split / "questions.jsonl"
    n = run(system, questions_path, args.out, max_retrieved=config.max_retrieved, workers=config.max_workers)
    print(json.dumps({"system": args.system, "split": args.split, "predictions": n}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
