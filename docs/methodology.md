# Methodology

This page specifies the benchmark precisely enough to re-derive every number.
The design rationale lives in
[`docs/superpowers/specs/2026-07-05-benchmark-rebuild-design.md`](superpowers/specs/2026-07-05-benchmark-rebuild-design.md).

## The file contract

Systems under test consume `questions.jsonl` and emit `predictions.jsonl`.

**questions.jsonl** — one task per line:

| field | type | meaning |
|---|---|---|
| `task_id` | str | opaque id, echo it back |
| `type` | str | `single_hop` \| `aggregation` \| `timeline` \| `unanswerable` |
| `question` | str | the question |

**predictions.jsonl** — one prediction per line, every task answered:

| field | type | meaning |
|---|---|---|
| `task_id` | str | must match a released task |
| `answer` | str | free text; explicit refusal when you believe the corpus lacks the answer |
| `citations` | [str] | document ids you claim support the answer |
| `retrieved` | [str] | your ranked retrieval list, ≤ 20 ids |

Document ids are the `doc_id` values of `aurora2424/Epstein-Files`
records (text-bearing rows only). Systems are not told a task's type beyond the field itself; note
that hiding the `type` field from your pipeline is fair play and closer to
production conditions.

## Scoring

Run: `python -m epstein_bench score predictions.jsonl --split full`

- **single_hop / timeline** — 1.0 iff the LLM judge (pinned model, prompt v1)
  says the answer states the same fact as the reference AND ≥1 cited doc is
  in the pooled gold set; else 0.0.
- **aggregation** — judge marks which gold items the answer includes and
  counts extra items. An included item *counts only if* a cited doc is in
  that item's supporting set (or the task's gold set). Score = item-level F1.
- **unanswerable** — 1.0 iff the judge classifies the answer as a
  refusal/abstention. A confident wrong answer scores 0 (that's the
  hallucination probe).
- **overall_cited_correctness** — unweighted macro-average over the four
  per-type scores. This is the leaderboard sort key.
- **retrieval diagnostics** — recall@5, recall@20, nDCG@10 of `retrieved`
  against pooled gold docs (answerable tasks only). Secondary columns; not
  part of the headline.

Judge failures during scoring count the affected task as 0 (fail closed) and
are reported in `judge_errors`.

## Splits

- `dev` — small fixed subset (source docs drawn from a seeded ~1K-doc subset,
  plus unanswerable top-ups). For iteration. Not leaderboard-eligible.
- `full` — the leaderboard split; retrieval is always over the full corpus.

`dataset/<version>/manifest.json` records counts and the sha256 of each
`questions.jsonl`; submissions pin that hash.

## Submissions

`python -m epstein_bench submit preds.jsonl --name "My System" --split full`
creates `submissions/<slug>/` with your predictions and metadata. Open a PR
adding it. CI (`.github/workflows/validate-submission.yml`):

1. checks bundle structure, dataset version, and questions-file hash;
2. verifies every task is answered;
3. **recomputes all scores from the predictions** and writes `scores.json`.

Self-reported scores are never used.

## Reproducing the dataset

`corpus → generate → verify → pool → finalize`, all seeded and config-pinned
(`src/epstein_bench/config.py`). LLM calls are disk-cached under
`build/llm_cache/`, so interrupted runs resume without re-spending. See the
[dataset card](../dataset/DATASET_CARD.md) for the verification gauntlet and
pooling design, including known limitations.
