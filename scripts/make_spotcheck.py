"""Produce a human spot-check file: a seeded random sample of released tasks
with their evidence, ready to review in a text editor.

    python scripts/make_spotcheck.py --n 100 --out build/spotcheck_v1.md

Verdict convention: mark each task OK / BAD / UNSURE in the checkbox line;
the observed error rate goes into dataset/DATASET_CARD.md release stats, and
BAD task_ids are retracted in a point release.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epstein_bench import DATASET_VERSION  # noqa: E402
from epstein_bench.config import Config  # noqa: E402
from epstein_bench.io_utils import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--split", default="full")
    parser.add_argument("--out", default="build/spotcheck_v1.md")
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    config = Config()
    tasks = list(
        read_jsonl(config.dataset_dir / DATASET_VERSION / args.split / "tasks.jsonl")
    )
    docs_by_id = {d["doc_id"]: d for d in read_jsonl(config.build_dir / "docs.jsonl")}
    sample = random.Random(args.seed).sample(tasks, min(args.n, len(tasks)))

    lines = [
        f"# Spot-check — dataset {DATASET_VERSION}, split {args.split}, "
        f"n={len(sample)}, seed={args.seed}",
        "",
        "For each task: is the question standalone, the answer correct per the",
        "evidence, and the evidence genuinely supporting? Mark one box.",
        "",
    ]
    for i, t in enumerate(sample, 1):
        lines += [
            "---",
            f"## {i}. `{t['task_id']}` ({t['type']})",
            f"**Q:** {t['question']}",
        ]
        if t["type"] == "aggregation":
            lines.append("**Gold items:**")
            lines += [f"- {it['item']}  (docs: {', '.join(it['doc_ids'])})" for it in t["items"]]
        elif t["type"] == "unanswerable":
            lines.append("**Gold:** refusal expected (answer not in corpus)")
        else:
            lines.append(f"**A:** {t['answer']}")
        for doc_id in (t.get("gold_docs") or t["source_doc_ids"])[:3]:
            doc = docs_by_id.get(doc_id)
            if doc:
                excerpt = " ".join(doc["text"][:600].split())
                lines.append(f"> **{doc_id}**: {excerpt}…")
        lines += ["", "Verdict: [ ] OK  [ ] BAD  [ ] UNSURE", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(sample)} tasks)")
    print(json.dumps({"by_type": {t: sum(1 for s in sample if s['type'] == t) for t in {s['type'] for s in sample}}}))


if __name__ == "__main__":
    main()
