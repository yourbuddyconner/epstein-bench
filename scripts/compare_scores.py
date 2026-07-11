"""Compare a committed scores.json against a CI-recomputed one.

The leaderboard is built from the scores.json committed in a submission PR, so
CI must prove that file honest: it rescores the predictions and fails the check
if the committed numbers differ from the recomputed ones beyond a per-field
tolerance.

    python scripts/compare_scores.py <committed.json> <recomputed.json>

Tolerances, by how each field is produced:
- deterministic arithmetic over the predictions (retrieval recall/nDCG,
  citation P/R, token/cost telemetry, counts): must match exactly (float eps);
- judge-derived aggregates (macro/micro correctness, CI bounds): the judge is a
  pinned model but not bit-deterministic, so a handful of verdict flips across
  ~1000 tasks is expected — small tolerance;
- judge-derived per-type scores and premise_id_rate: tiny strata (dossier n=7,
  false_premise n=38) move a lot on one flip — wider tolerance. Inflating a
  tiny type to game the macro is still caught by the macro tolerance.
"""

from __future__ import annotations

import json
import re
import sys

EPS = 1e-6  # deterministic floats (rounding only)
JUDGE_TOL = 0.02  # judge-derived aggregates
SMALL_STRATA_TOL = 0.15  # judge-derived per-type / small-n rates

# path-pattern -> tolerance; first match wins, EPS otherwise
TOLERANCES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\.per_type(_uncited)?\."), SMALL_STRATA_TOL),
    (re.compile(r"\.premise_id_rate$"), SMALL_STRATA_TOL),
    (re.compile(r"\.overall_(cited|uncited)_correctness(_ci95|_micro)?"), JUDGE_TOL),
]
# expected to differ between runs; never a mismatch
IGNORE_KEYS = {"judge_errors"}


def _tolerance(path: str) -> float:
    for pattern, tol in TOLERANCES:
        if pattern.search(path):
            return tol
    return EPS


def _diff(committed, recomputed, path: str, problems: list[str]) -> None:
    if path.rsplit(".", 1)[-1] in IGNORE_KEYS:
        return
    if isinstance(committed, dict) and isinstance(recomputed, dict):
        for k in sorted(set(committed) | set(recomputed)):
            if k not in committed or k not in recomputed:
                problems.append(f"{path}.{k}: present in only one file")
            else:
                _diff(committed[k], recomputed[k], f"{path}.{k}", problems)
        return
    if isinstance(committed, list) and isinstance(recomputed, list):
        if len(committed) != len(recomputed):
            problems.append(f"{path}: list lengths differ")
            return
        for i, (c, r) in enumerate(zip(committed, recomputed)):
            _diff(c, r, f"{path}[{i}]", problems)
        return
    numeric = (
        isinstance(committed, (int, float))
        and isinstance(recomputed, (int, float))
        and not isinstance(committed, bool)
        and not isinstance(recomputed, bool)
    )
    if numeric:
        tol = _tolerance(path)
        if abs(float(committed) - float(recomputed)) > tol:
            problems.append(
                f"{path}: committed {committed} vs recomputed {recomputed} "
                f"(tolerance {tol})"
            )
        return
    if committed != recomputed:
        problems.append(f"{path}: committed {committed!r} vs recomputed {recomputed!r}")


def main() -> int:
    committed_path, recomputed_path = sys.argv[1], sys.argv[2]
    committed = json.load(open(committed_path))
    recomputed = json.load(open(recomputed_path))
    problems: list[str] = []
    _diff(committed, recomputed, "scores", problems)
    if problems:
        print(f"MISMATCH: committed {committed_path} != recomputed scores:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("scores match (within judge tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
