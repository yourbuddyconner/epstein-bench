"""Command-line pipeline: corpus -> generate -> verify -> pool -> finalize -> score.

Every stage reads/writes JSONL artifacts under ``build/`` and is independently
re-runnable; LLM calls are disk-cached, so re-running a crashed stage resumes
rather than re-spending.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import Config
from .io_utils import read_jsonl
from .llm import LLM


def _config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    if getattr(args, "limit", None):
        config.doc_limit = args.limit
    if getattr(args, "target", None):
        config.target_tasks = args.target
    config.ensure_dirs()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="epstein_bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="wide scan: cache text + entity mentions per shard")
    p.add_argument("--shards", type=int, help="number of parquet shards (default all)")

    sub.add_parser(
        "select",
        help="entity-complete corpus from the scan cache (targets + backbone)",
    )

    p = sub.add_parser("corpus", help="load, screen, chunk, and index the corpus")
    p.add_argument("--limit", type=int, help="cap number of documents (dev runs)")

    p = sub.add_parser("generate", help="draft candidate tasks (oversampled)")
    p.add_argument("--target", type=int, help="final task target (default 1000)")

    sub.add_parser("verify", help="run the verification gauntlet")
    sub.add_parser("pool", help="build pooled retrieval ground truth")

    p = sub.add_parser("finalize", help="cut versioned dev/full splits")
    p.add_argument("--target", type=int, help="final task target (default 1000)")

    p = sub.add_parser("score", help="score a predictions file")
    p.add_argument("predictions")
    p.add_argument("--split", default="full", choices=["dev", "full"])

    p = sub.add_parser("submit", help="build a leaderboard submission bundle")
    p.add_argument("predictions")
    p.add_argument("--name", required=True, help="system name")
    p.add_argument("--split", default="full", choices=["dev", "full"])
    p.add_argument("--description", default="")
    p.add_argument("--out", default="submissions")

    p = sub.add_parser("validate", help="validate + rescore a submission bundle (CI)")
    p.add_argument("bundle")

    args = parser.parse_args(argv)
    config = _config_from_args(args)
    llm = LLM(config)

    if args.command == "scan":
        from .scan import scan_corpus

        if getattr(args, "shards", None):
            config.scan_shards = args.shards
        stats = scan_corpus(config)
    elif args.command == "select":
        from .scan import select_corpus

        stats = select_corpus(config, llm)
    elif args.command == "corpus":
        from .corpus import build_corpus

        stats = build_corpus(config, llm)
    elif args.command == "generate":
        from .generate import generate_candidates

        stats = generate_candidates(config, llm)
    elif args.command == "verify":
        from .verify import verify_candidates

        stats = verify_candidates(config, llm)
    elif args.command == "pool":
        from .pool import pool_tasks

        stats = pool_tasks(config, llm)
    elif args.command == "finalize":
        from .finalize import finalize_dataset

        stats = finalize_dataset(config)
    elif args.command == "score":
        from . import DATASET_VERSION
        from .score import score_predictions

        tasks = list(
            read_jsonl(config.dataset_dir / DATASET_VERSION / args.split / "tasks.jsonl")
        )
        predictions = list(read_jsonl(args.predictions))
        stats = score_predictions(config, llm, tasks, predictions)
    elif args.command == "submit":
        from .submit import build_bundle

        bundle = build_bundle(
            config, args.predictions, args.name, args.out, args.split, args.description
        )
        stats = {"bundle": str(bundle)}
    elif args.command == "validate":
        from .submit import validate_bundle

        stats = validate_bundle(config, llm, args.bundle)
    else:  # pragma: no cover - argparse enforces choices
        return 2

    json.dump(stats, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
