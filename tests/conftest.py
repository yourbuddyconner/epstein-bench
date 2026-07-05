import os
import sys
from pathlib import Path

import pytest

os.environ["EPSTEIN_BENCH_STUB_LLM"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epstein_bench import llm as llm_mod  # noqa: E402
from epstein_bench.config import Config  # noqa: E402
from epstein_bench.llm import LLM  # noqa: E402

# Fixture corpus: clean docs are built mostly from words in the built-in
# fallback wordlist so the quality screen classifies them 'clean' regardless
# of whether the NLTK words corpus is installed.
_CLEAN_TEMPLATE = (
    "From: Alice Example. Sent: January {day} 2015. To: Bob Sample. "
    "Subject: the meeting about the house committee work. Dear Bob, thank you "
    "for the email about the meeting. Please call me before the meeting so "
    "that we can go over all of the work that must be made before the house "
    "committee will see it. I know that there is much more work to do and we "
    "should take time this year to make good work of it for the people. "
    "It would be good if you could come down before the end of the year so "
    "that we can see where the work stands and what more must be made new. "
    "Best regards, Alice Example."
)


def fixture_rows() -> list[dict]:
    rows = [
        {
            "doc_id": f"CLEAN-{i:03d}",
            "file_name": f"CLEAN-{i:03d}.pdf",
            "file_type": "pdf",
            "text_content": _CLEAN_TEMPLATE.format(day=10 + i),
        }
        for i in range(8)
    ]
    rows.append({"doc_id": "SHORT-001", "file_type": "pdf", "text_content": "too short"})
    rows.append(
        {
            "doc_id": "NOISY-001",
            "file_type": "pdf",
            "text_content": ("ÿþ□zx qq " * 60) + "the of and to in a is that",
        }
    )
    # media rows carry no text and must be skipped entirely
    rows.append({"doc_id": "IMG-001", "file_type": "image", "text_content": None})
    return rows


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.build_dir = tmp_path / "build"
    cfg.dataset_dir = tmp_path / "dataset"
    cfg.cache_dir = tmp_path / "build" / "llm_cache"
    cfg.stub_llm = True
    cfg.target_tasks = 8
    cfg.entity_min_count = 3
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def llm(config) -> LLM:
    return LLM(config)


@pytest.fixture(autouse=True)
def clean_stub_overrides():
    llm_mod.STUB_OVERRIDES.clear()
    yield
    llm_mod.STUB_OVERRIDES.clear()
