# Epstein Bench v1 — First-Principles Design

**Date:** 2026-07-05
**Status:** Approved by Conner (brainstorming session; supersedes the earlier "verification-first rebuild" draft after an adversarial review)
**Goal:** A credible public RAG benchmark on the Epstein Files corpus. Not a framework, not a library — a dataset, a file contract, and a scorer.

## 1. Premise

The corpus (`aurora2424/Epstein-Files`, the full public release; originally scoped to `tensonaut/EPSTEIN_FILES_20K`) is the asset: millions of real, noisy, OCR'd, entity-rich documents — the kind of haystack enterprise RAG actually faces and Wikipedia-derived benchmarks don't test. Everything in this design exists to answer one question trustworthily: **how well does a RAG system answer real questions over this corpus, with evidence?**

The previous codebase failed because its ground truth was unverified and circular (heuristic answer spans, source-chunk-as-gold retrieval labels, no answer checking). The previous *plan* failed adversarial review because it anchored on existing code: a pluggable Python framework nobody needs, bridge-style multi-hop tasks nobody asks, paraphrase tasks that test nothing, and 1,500 lines of unused GraphRAG. All of that is deleted, not preserved.

## 2. Decisions

| Question | Decision |
|---|---|
| Ambition | Credible public benchmark others can use |
| Corpus scale | Tiered: dev split + full split; retrieval always over the full text corpus |
| Participation | Self-run, submit predictions via PR; CI recomputes scores |
| Verification | Automated LLM gauntlet on every task + human spot-check; pass rates published |
| Budget | Cheap-first (~$50–150): mini-tier model for bulk work, strong model only for final adjudication |

## 3. The contract

A system under test never imports our code. The interface is two JSONL files:

**Input — `questions.jsonl`** (per line):
```json
{"task_id": "...", "question": "...", "type": "single_hop|aggregation|timeline|unanswerable"}
```

**Output — `predictions.jsonl`** (per line):
```json
{"task_id": "...", "answer": "...", "citations": ["doc_id", ...], "retrieved": ["doc_id", ...]}
```
- `answer`: free text; for unanswerable tasks the expected behavior is an explicit refusal.
- `citations`: doc IDs the system claims support its answer (the subset of retrieved evidence it actually relied on).
- `retrieved`: the ranked retrieval list (for recall/nDCG), up to 20 IDs.

One command scores it: `python -m epstein_bench score predictions.jsonl --split full`. This contract is language-agnostic and is exactly what CI recomputes on submission PRs.

## 4. Corpus preparation

- **Quality screen** before generation: heuristic pass (garbage-character ratio, dictionary-word ratio, length) plus cheap-model readability check on borderline docs → `clean` / `degraded` / `garbage`. Tasks are generated **only from clean text**; `degraded` docs stay in the retrieval corpus as natural distractors; `garbage` is excluded. Noise lives in the haystack, never in the answer key.
- **Chunking:** ~512-token chunks with overlap for retrieval scoring granularity; doc-level IDs are the citation unit (systems cite documents, not our chunk boundaries).
- **Entity alias index:** a lightweight index of person/org aliases (built once, cheap model + heuristics) used for bounding aggregation tasks (§5) and for disambiguation-aware generation.

## 5. Task types

Generation is **fact-first**: extract verified atomic facts from clean text (each fact carries all supporting docs found via pooling, §7), then write the question an investigator would ask against the fact — not a reading-comprehension item about one paragraph.

| Type | Share | Definition | Gold |
|---|---|---|---|
| `single_hop` | ~50% | Factoid question against one atomic fact (name, date, amount, event). Must name concrete entities; no deictic references. | Short answer string + supporting doc set |
| `aggregation` | ~20% | Bounded list/count questions ("Which people are named in correspondence with X about Y?"). Bounded = scoped to an entity/date range where candidate docs can be enumerated via the alias index and each judged. | Item set with per-item supporting docs |
| `timeline` | ~15% | Temporal ordering/range questions over 2+ documents ("Over what period…", "What happened between…"). | Answer + supporting doc set |
| `unanswerable` | ~15% | Plausible questions whose answer provably isn't in the corpus (perturbed entities/facts, verified absent via search). | Expected refusal; empty doc set |

Explicitly rejected task types: bridge-style multi-hop (synthetic compositions nobody asks), paraphrase/typo variants (no discriminative signal), global sensemaking (unverifiable gold).

## 6. Verification gauntlet

Every candidate task passes all stages or is discarded with a logged failure reason. Generation oversamples ~3× the target count. Cheap model for stages 1–3, strong model for stage 4.

1. **Standalone check:** question interpretable with no source document in view; concrete entities; no boilerplate-derived content.
2. **Answerability check** (fresh context): given the gold docs, a different-from-generator prompt must produce the reference answer (semantic match, token-F1 sanity floor). For aggregation: recover the item set.
3. **Necessity check:** closed-book + random-distractor attempts must *fail*; for multi-doc types, any single gold doc alone must be insufficient.
4. **Adjudication** (strong model): pass/fail with failure category (ambiguous / wrong / not grounded / trivial).

**Human spot-check:** random sample of a few hundred final tasks reviewed by hand; observed error rate published in the dataset card.

## 7. Retrieval ground truth by pooling

TREC-style pooled judgments replace source-chunk circularity:

1. Per task, run three diverse retrievers over the full corpus (BM25, dense, hybrid); union of top-20 each, source docs force-included. For aggregation tasks, the pool additionally includes all alias-index hits for the bounding entity.
2. Cheap model judges each pooled doc: `supports answer` / `partial` / `irrelevant`.
3. Gold set = all `supports` docs (duplicated facts get multiple golds).
4. Strong-model stability re-check on a sample; unstable tasks dropped.
5. Docs state plainly: relevance sets are pooled, not exhaustive; pool composition is versioned.

## 8. Splits and versioning

- `dev` — ~150 tasks from a fixed ~1K-doc subset (retrieval still over the full corpus). For iteration; not leaderboard-eligible.
- `full` — ~1,000 verified tasks spanning the corpus. Leaderboard split.
- Releases are versioned (`v1.0`, `v1.1`…); every task carries provenance (source docs, generator model, verification verdicts, pipeline version). Bad tasks are retracted in point releases, never silently edited.

## 9. Scoring

Headline metric: **cited answer correctness** — an answer scores only if (a) it matches the reference (LLM judge, pinned model + published prompt; token-F1 reported as secondary) **and** (b) at least one cited doc is in the gold supporting set. Per type:

- `single_hop` / `timeline`: cited correctness (binary per task).
- `aggregation`: item-level precision/recall/F1, an item counting only with a supporting citation.
- `unanswerable`: refusal accuracy (refused = correct; any confident answer = hallucination).
- Retrieval diagnostics (secondary columns): recall@5, recall@20, nDCG@10 against pooled gold sets.

Leaderboard sorts by overall cited answer correctness (macro-average over types). No composite weighted score. Judge model/prompt version is part of the release; changing the judge is a new benchmark version.

## 10. Submissions and leaderboard

- PRs contain a results bundle: `predictions.jsonl` + system description + dataset version hash. GitHub Action validates completeness/version and **recomputes all scores from predictions** — submitters never submit scores.
- Launch baselines (run by us): BM25-only, dense-only, hybrid + rerank, and **closed-book** (no retrieval) — the closed-book row is public evidence the tasks require retrieval.
- The existing leaderboard resets; old entries and data files are deleted.

## 11. Repo reshape

The repo becomes small:

```
dataset/            # versioned splits + DATASET_CARD.md
src/epstein_bench/  # generate, verify, pool, score, submit — a pipeline, not a framework
baselines/          # the four reference systems
docs/               # leaderboard site, methodology
tests/
```

Deleted outright (git history preserves them): `src/auepora_eval/` (framework, metric zoo, taskgen), `src/graphrag/`, stale data files (`epstein_bench_200docs.jsonl`, the `my_bench.jsonl`-based leaderboard entry), aspirational spec-as-docs. `DATASET_CARD.md` documents methodology, verification pass rates, known limitations (pooled relevance, synthetic questions, OCR noise), and an ethics note (public-record documents; no new personal information synthesized; tasks generated only from already-public text).

## 12. Reproducibility and error handling

- All generation/verification seeded and config-pinned; one command re-runs the pipeline end-to-end; intermediate artifacts (facts, candidates, verdicts, pool judgments) cached to disk so runs resume after crashes.
- LLM failures: bounded retries; persistent failure → candidate discarded and logged, never silently passed. Verification fails closed. CI validation fails closed with a readable reason on the PR.

## 13. Testing

- Unit tests: quality screen (fixture docs), each gauntlet stage's accept/reject (stub LLM), pool aggregation, scorer against hand-computed fixtures, submission validation.
- End-to-end smoke test on ~10 fixture docs with a stub LLM, runnable in CI without API keys.

## 14. Cost envelope

~3,500 candidates × a few mini-tier calls (facts + generation + stages 1–3) plus strong-model adjudication on ~1,200 survivors and stability samples ≈ within the $50–150 target. All stages batched and resumable.

## 15. Roadmap (explicitly not v1)

- Human-verified `core` split.
- Held-out private test set / hosted scoring.
- Harder task families if verifiable gold can be constructed (cross-document contradiction detection, entity disambiguation stress set).
