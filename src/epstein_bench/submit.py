"""Submission bundles: build one from a predictions file, validate one in CI.

A leaderboard PR contains ``submissions/<name>/`` with:
- ``predictions.jsonl`` — the system's raw predictions
- ``metadata.json``     — system name/description, dataset version, questions hash

Scores are ALWAYS recomputed from predictions (by ``validate``, run in CI);
submitters never submit scores.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from . import DATASET_VERSION
from .config import Config
from .io_utils import read_jsonl
from .llm import LLM
from .score import score_predictions


def _questions_sha256(config: Config, split: str) -> str:
    p = config.dataset_dir / DATASET_VERSION / split / "questions.jsonl"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_bundle(
    config: Config,
    predictions_path: str | Path,
    system_name: str,
    out_dir: str | Path,
    split: str = "full",
    description: str = "",
) -> Path:
    slug = re.sub(r"[^a-z0-9-]+", "-", system_name.lower()).strip("-") or "system"
    bundle = Path(out_dir) / slug
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy(predictions_path, bundle / "predictions.jsonl")
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "system_name": system_name,
                "description": description,
                "split": split,
                "dataset_version": DATASET_VERSION,
                "questions_sha256": _questions_sha256(config, split),
            },
            indent=2,
        )
    )
    return bundle


def validate_bundle(config: Config, llm: LLM, bundle_dir: str | Path) -> dict:
    """Fail-closed validation + score recomputation. Raises ValueError on any problem."""
    bundle = Path(bundle_dir)
    meta_path = bundle / "metadata.json"
    preds_path = bundle / "predictions.jsonl"
    if not meta_path.exists() or not preds_path.exists():
        raise ValueError("bundle must contain metadata.json and predictions.jsonl")

    meta = json.loads(meta_path.read_text())
    for key in ("system_name", "split", "dataset_version", "questions_sha256"):
        if not meta.get(key):
            raise ValueError(f"metadata.json missing '{key}'")
    if meta["dataset_version"] != DATASET_VERSION:
        raise ValueError(
            f"dataset_version {meta['dataset_version']} != current {DATASET_VERSION}"
        )
    split = meta["split"]
    actual_hash = _questions_sha256(config, split)
    if actual_hash != meta["questions_sha256"]:
        raise ValueError("questions_sha256 does not match the released questions file")
    # the manifest is the released pin: the working-tree questions file must
    # match it, so a PR that tampers with the dataset cannot self-consistently
    # validate against its own modified questions
    manifest_path = config.dataset_dir / DATASET_VERSION / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pinned = manifest["splits"][split]["questions_sha256"]
    if actual_hash != pinned:
        raise ValueError(
            f"questions.jsonl ({split}) does not match the released manifest pin "
            f"{pinned[:12]}… — the dataset files have been modified"
        )

    predictions = list(read_jsonl(preds_path))
    for i, p in enumerate(predictions):
        if "task_id" not in p or "answer" not in p:
            raise ValueError(f"prediction line {i + 1} missing task_id/answer")
        if len(p.get("citations") or []) > config.max_retrieved:
            raise ValueError(
                f"prediction line {i + 1} has more than {config.max_retrieved} citations"
            )
        if len(p.get("retrieved") or []) > config.max_retrieved:
            raise ValueError(
                f"prediction line {i + 1} has more than {config.max_retrieved} retrieved ids"
            )

    tasks = list(
        read_jsonl(config.dataset_dir / DATASET_VERSION / split / "tasks.jsonl")
    )
    report = score_predictions(config, llm, tasks, predictions)
    if report["judge_errors"]:
        # in CI a partially-judged report must not become a green check: rerun
        # (judge calls are cached, so a rerun only re-attempts the failures)
        raise ValueError(
            f"scoring hit {report['judge_errors']} judge error(s); rerun validation"
        )
    report["system_name"] = meta["system_name"]
    report["split"] = split
    report["dataset_version"] = DATASET_VERSION
    (bundle / "scores.json").write_text(json.dumps(report, indent=2))
    return report
