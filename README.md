# Epstein Bench

<p align="center">
  <img src="mascot.png" alt="Epstein Bench Mascot" width="220" />
</p>

A RAG benchmark over the **Epstein Files** — the full public release
(~4.1M files; the ~1M+ text-bearing documents form the retrieval corpus):
OCR'd, noisy, entity-rich. Wikipedia-derived benchmarks test
retrieval over clean encyclopedic text; this one tests what enterprise RAG
actually faces: messy scans, duplicated emails, boilerplate, and facts
scattered across documents.

**The benchmark is a dataset, a file contract, and a scorer — not a framework.**
Your system never imports this code.

## The contract

Read `dataset/v1.0/<split>/questions.jsonl`:

```json
{"task_id": "...", "question": "...", "type": "single_hop|aggregation|timeline|unanswerable"}
```

Emit `predictions.jsonl`:

```json
{"task_id": "...", "answer": "...", "citations": ["doc_id"], "retrieved": ["doc_id", "..."]}
```

- `answer` — free text; for unanswerable tasks the correct behavior is an explicit refusal
- `citations` — document ids your system claims support the answer
- `retrieved` — your ranked retrieval list (max 20), for retrieval diagnostics

Score it:

```bash
python -m epstein_bench score predictions.jsonl --split full
```

## Headline metric: cited answer correctness

An answer scores only if it matches the reference (pinned LLM judge, published
prompt) **and** at least one cited document genuinely supports it. Per type:
binary cited correctness (single_hop, timeline), item-level P/R/F1 with
citation requirement (aggregation), refusal accuracy (unanswerable).
Retrieval diagnostics: recall@5/20, nDCG@10 against pooled gold sets.

## Why trust the ground truth

Every task in a release has survived a four-stage verification gauntlet —
standalone interpretability, independent re-answering from the gold docs,
closed-book/distractor necessity checks, and strong-model adjudication —
plus TREC-style pooled relevance judgments across three diverse retrievers,
with a stability re-check. Verification pass rates and known limitations are
published in [`dataset/DATASET_CARD.md`](dataset/DATASET_CARD.md), and every
task carries full provenance. Methodology details:
[`docs/methodology.md`](docs/methodology.md).

## Quick start

```bash
conda create -n epstein-bench python=3.11 && conda activate epstein-bench
pip install -e ".[dev]"
cp env.example .env   # add OPENAI_API_KEY (needed for scoring's LLM judge)
```

Run a reference baseline on the dev split:

```bash
python -m epstein_bench corpus                 # one-time: build corpus artifacts
python baselines/run_baseline.py --system hybrid --split dev --out preds.jsonl
python -m epstein_bench score preds.jsonl --split dev
```

Baselines: `bm25`, `dense`, `hybrid`, and `closed_book` (no retrieval — public
evidence the tasks require it).

## Submitting to the leaderboard

1. Run your system on `dataset/v1.0/full/questions.jsonl`.
2. `python -m epstein_bench submit predictions.jsonl --name "My System" --split full`
3. Open a PR adding the generated `submissions/<name>/` directory.

CI recomputes all scores from your predictions — submitted scores are never
trusted. The `dev` split is for iteration and is not leaderboard-eligible.

## Regenerating the dataset

The pipeline is seeded, config-pinned, and resumable (all LLM calls are
disk-cached):

```bash
python -m epstein_bench corpus
python -m epstein_bench generate --target 1000
python -m epstein_bench verify
python -m epstein_bench pool
python -m epstein_bench finalize
```

## Repository layout

```
dataset/            versioned task splits + DATASET_CARD.md
src/epstein_bench/  the pipeline: corpus, generate, verify, pool, score, submit
baselines/          reference systems (bm25 / dense / hybrid / closed_book)
docs/               methodology + leaderboard site
tests/              unit + end-to-end smoke tests (stub LLM, no API key needed)
```

## Ethics

The corpus consists of documents released by government bodies and courts
(`aurora2424/Epstein-Files`). Tasks are generated only from already-public
text; no new personal information is synthesized. The benchmark's purpose is
retrieval research on realistic noisy corpora.

## Citation

```bibtex
@software{epstein_bench_2026,
  title  = {Epstein Bench: a verified RAG benchmark over the public Epstein Files},
  author = {Conner Swann},
  year   = {2026},
  url    = {https://github.com/yourbuddyconner/epstein-bench}
}
```
