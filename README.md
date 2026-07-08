# Epstein Bench

<p align="center">
  <img src="mascot.png" alt="Epstein Bench Mascot" width="220" />
</p>

**Millions of scanned, garbled, redaction-strewn documents. Can your AI find the
one sentence that answers the question?**

The public **Epstein Files** run to millions of released records. Every other
retrieval benchmark quizzes AI on clean Wikipedia; the real world looks nothing
like this: OCR wreckage, near-duplicate emails, endless legalese, and the one
fact you need buried somewhere in the pile. Epstein Bench distills the release
into a benchmark: a retrieval corpus of about 84,000 text documents and 1,038
questions, answerable only by finding the right document and citing it. Live
leaderboard and example questions at **[epsteinbench.com](https://epsteinbench.com)**.

Sample tasks: *Who did Epstein ask to find him "the best codebreaker, NSA type"?*
· *What did Steve Bannon email him about "real power"?* · *Reconstruct the
documented timeline of a given person's contacts with Epstein.* Every answer is
one sentence hiding somewhere in the corpus.

**Epstein Bench is a dataset, a file contract, and a scorer.** Your system never
imports this code. It reads `questions.jsonl` and writes `predictions.jsonl`.

> These are public records released by U.S. courts and Congress. Appearing in
> the files means appearing in someone's email, calendar, or financial records.
> It is not an accusation. The benchmark measures retrieval, not guilt.

## The contract

Read `dataset/v1.0/<split>/questions.jsonl`:

```json
{"task_id": "...", "question": "...", "type": "single_hop|aggregation|timeline|dossier|unanswerable|false_premise"}
```

Emit `predictions.jsonl`:

```json
{"task_id": "...", "answer": "...", "citations": ["doc_id"], "retrieved": ["doc_id", "..."]}
```

- `answer`: free text; for `unanswerable` and `false_premise` tasks the correct behavior is an explicit refusal (reject the premise, do not answer the follow-up)
- `citations`: document ids your system claims support the answer
- `retrieved`: your ranked retrieval list (max 20), for retrieval diagnostics

Score it:

```bash
python -m epstein_bench score predictions.jsonl --split full
```

## Headline metric: cited answer correctness

An answer scores only if it matches the reference (pinned strong-model judge,
published prompt) **and** at least one cited document genuinely supports it (only
the first few citations count, so dumping the retrieval list cannot game it).
Per type: binary cited correctness (single_hop, timeline), item-level P/R/F1
with citation requirement (aggregation, dossier), rejection accuracy
(unanswerable, false_premise). The macro headline ships with a bootstrap 95%
confidence interval and a task-weighted micro average. Diagnostics: citation
precision/recall, false-premise identification rate, and retrieval recall@5/20,
nDCG@10 against pooled gold sets.

## Why trust the ground truth

Every task in a release has survived a four-stage verification gauntlet
(standalone interpretability, independent re-answering from the gold docs,
closed-book/distractor necessity checks, and strong-model adjudication),
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

Baselines: `bm25`, `dense`, `hybrid`, and `closed_book` (no retrieval, public
evidence the tasks require it).

## Submitting to the leaderboard

1. Run your system on `dataset/v1.0/full/questions.jsonl`.
2. `python -m epstein_bench submit predictions.jsonl --name "My System" --split full`
3. Open a PR adding the generated `submissions/<name>/` directory.

CI recomputes all scores from your predictions, submitted scores are never
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
