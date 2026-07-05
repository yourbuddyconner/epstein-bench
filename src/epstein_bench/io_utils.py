"""JSONL + concurrency helpers used by every pipeline stage."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

_T = TypeVar("_T")
_R = TypeVar("_R")


def parallel_map(
    fn: Callable[[_T], _R], items: list[_T], max_workers: int
) -> list[_R]:
    """Ordered thread-pool map. Per-item error handling belongs inside ``fn``
    (pipeline stages fail closed per item, never crash the whole stage)."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(fn, items))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]] | Iterator[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
