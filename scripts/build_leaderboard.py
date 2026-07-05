"""Rebuild docs/leaderboard.json from validated submission bundles.

Run after merging a submission PR (CI wrote scores.json into the bundle):
    python scripts/build_leaderboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMISSIONS = REPO / "submissions"
OUT = REPO / "docs" / "leaderboard.json"


def main() -> None:
    entries = []
    if SUBMISSIONS.exists():
        for scores_path in sorted(SUBMISSIONS.glob("*/scores.json")):
            report = json.loads(scores_path.read_text())
            if report.get("split") != "full":
                continue  # dev runs are not leaderboard-eligible
            entries.append(report)
    entries.sort(key=lambda r: -r.get("overall_cited_correctness", 0.0))
    OUT.write_text(json.dumps({"entries": entries}, indent=2))
    print(f"wrote {OUT} with {len(entries)} entries")


if __name__ == "__main__":
    main()
