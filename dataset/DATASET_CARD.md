# Epstein Bench Dataset Card

**Current release:** `v1.0` (2026-07-07). This card documents the
methodology; the *Release statistics* section is updated whenever a version
ships.

## Source corpus

- `aurora2424/Epstein-Files` (Hugging Face): the full public Epstein Files
  release, ~4.1M rows (340GB including raw media bytes). Only the text
  columns are consumed, via parquet column projection; rows without
  `text_content` (images, audio, video) are skipped. Emails, depositions,
  flight logs, scanned letters. The media columns (`image`, `audio`,
  `video`, `online_url`) open the door to multimodal task families later;
  v1 is text-only.
- Documents are quality-screened into `clean` / `degraded` / `garbage`.
  Tasks are generated **only from clean text**; degraded documents remain in
  the retrieval corpus as natural distractors; garbage is excluded entirely.

## Task types

| type | gold | scored as |
|---|---|---|
| `single_hop` | short answer + supporting docs | cited answer correctness (binary) |
| `aggregation` | item list, per-item supporting docs | item-level P/R/F1, citation-gated |
| `timeline` | short answer + supporting docs (≥2 required) | cited answer correctness (binary) |
| `dossier` | dated event list for a notable person, per-item docs | item-level P/R/F1, citation-gated |
| `unanswerable` | none (refusal expected) | refusal accuracy |
| `false_premise` | none (rejection expected); `false_element` records the fabrication | rejection accuracy + premise-identification diagnostic |

Corpus selection is **entity-complete**: a wide scan indexes entity mentions
across the source dataset, an LLM notability check picks target people (public
figures only; entities appearing in more than `max_entity_docs` documents are
excluded as impractically pervasive), and the corpus is all documents
mentioning any target plus a seeded random backbone. Single-hop facts are
**salience-filtered** (newsworthiness ≥3/5: notable people, money flows,
meetings/travel, legal exposure, never speculation; facts must be
document-stated).

Generation is fact-first: atomic facts are extracted from clean documents and
questions are written against the fact, in investigator phrasing. Aggregation
questions are **bounded**, scoped to an entity whose candidate documents are
enumerable via an alias index, because unbounded "list all X" gold sets
cannot be verified at corpus scale.

## Verification

Every shipped task passed all of:

1. **Standalone**: interpretable without the source document (concrete
   entities, no deixis, no boilerplate targets).
2. **Answerability**: an independent prompt, shown the gold documents,
   recovers the reference answer (semantic match + token-F1 floor; ≥80% item
   recovery for aggregation).
3. **Necessity**: closed-book and random-distractor attempts fail; for
   multi-document types, no single gold document suffices.
4. **Adjudication**: a stronger model passes/fails with a category.

Unanswerable tasks run stages 1 and 4, plus a generation-time absence check
(top BM25 hits confirmed non-answering). All rejections are logged with the
failing stage (`build/rejected.jsonl`).

**False-premise tasks** are anchored on entity-complete targets (so "no document
supports this" is bounded) and fabricate an interaction between the target and a
prominent outside figure, rotated across a diverse pool. They skip stages 2–3
and run a generation-time absence check plus a two-stage adjudication: a neutral
support check that drops any premise the on-topic documents actually support
(catching premises that merely perturb a detail of a real meeting), and a
quality check for plausibility and that the wording does not reveal the premise
is false.

## Retrieval ground truth (pooled)

Gold document sets come from TREC-style pooling: the union of top-20 results
from three diverse retrievers (BM25, dense embeddings, hybrid RRF) plus the
source documents, each judged supports/partial/irrelevant. Gold = all
'supports' documents. A sample is re-judged by the strong model; tasks with
supports↔irrelevant flips are dropped as unstable.

**Limitation:** pooled relevance sets are not exhaustive. A document outside
the pool that happens to state the answer will be scored as non-gold. Pool
composition is versioned with each release.

## Models (pinned per release)

| role | model |
|---|---|
| generation + gauntlet stages 1-3 + pool judging | `gpt-4o-mini-2024-07-18` |
| adjudication + pool stability re-check | `gpt-4o-2024-08-06` |
| scoring judge (prompt v1) | `gpt-5.5-2026-04-23` |

The scoring judge is a strong model because correctness judging approaches human
agreement at that tier; generation and gauntlet filtering tolerate the cheaper
model. Changing the scoring judge or its prompt is a new benchmark version;
scores across versions are not comparable.

## Release statistics

### v1.0 (2026-07-07)

- **Corpus:** entity-complete selection over the full source dataset, a wide
  scan of all 634 parquet shards (~1.38M text-bearing docs) indexed entity
  mentions; 40 notable target people were chosen (LLM notability check) and the
  retrieval corpus is every document mentioning a target plus a 30,000-doc
  random backbone, for a corpus of 83,810 documents and 159,564 chunks.
- **Tasks:** 1,038 (`full`): 823 single_hop / 111 aggregation / 27 timeline /
  7 dossier / 32 unanswerable / 38 false_premise. `dev`: 50. Multi-document
  types (timeline, dossier) survive verification at low rates, so they remain a
  small share.
- **Gauntlet:** 1,098 of 4,034 candidates passed (27%) in the original run.
  Rejections: 1,528 standalone, 1,080 answerability, 786 adjudication, 229
  necessity. The 38 `false_premise` tasks were added from 39 candidates that
  passed the two-stage adjudication (1 dropped as non-standalone; earlier
  candidates dropped when the corpus supported the premise or the premise merely
  perturbed a real meeting).
- **Pooling:** 1,018 of 1,098 kept; 74 dropped as unstable under strong-model
  re-judging, 6 as source-not-supportive. One non-person dossier target was
  retracted, leaving 1,000 tasks, plus 38 `false_premise` (no pooling; empty
  gold set), for 1,038 released.
- **Reference baselines** (cited correctness, 95% CI, micro; scored by
  `gpt-5.5-2026-04-23`): agentic-sonnet-5 0.553 [0.52, 0.59] / 0.540,
  agentic-opus-4-8 0.494 [0.46, 0.53] / 0.467, hybrid 0.494 [0.46, 0.54] /
  0.489, bm25 0.492 [0.45, 0.53] / 0.482, dense 0.440 [0.40, 0.47] / 0.398,
  closed_book 0.328 [0.32, 0.33] / 0.066, parametric 0.304 [0.28, 0.32] / 0.062.
  Among one-shot retrievers hybrid and bm25 tie; dense trails. closed_book and
  parametric score 0.000 on every retrieval-requiring type — their macro is
  entirely rejection accuracy, which the micro (0.066 / 0.062) exposes. The
  Sonnet-5 agent tops the board; the pricier Opus-4.8 agent merely ties one-shot
  hybrid. **Cost/tokens** (agents only): Sonnet $0.053/task (~$55/split, ~14.8k
  tok), Opus $0.114/task (~$119, ~19.6k tok). **False-premise:** every system
  refuses at 0.95–1.00 (headline saturates), but the premise-identification
  diagnostic separates the agents (1.0 — both name the fabrication) from every
  non-agent baseline (0.0). **Parametric knowledge probe:** single_hop uncited
  0.041 (vs closed_book 0.016): the generation-time model answers ~4% of
  single-hop facts from training weights alone, a small but measurable
  contamination signal.
- **Spot-check:** all 7 dossiers reviewed by hand (real public figures such as
  Steven Sinofsky, Reid Weingarten, Martin Weinberg, and Lesley Groff, with
  correctly dated, document-grounded events) plus a read of ~20
  single-hop/aggregation triples against source text: no clear errors;
  questions are standalone and name concrete entities. Automated grounding
  (gold answer present in a gold document, verbatim or ≥60% token overlap):
  single_hop 99.0% (815/823), timeline 96.3% (26/27); the residual are
  date-format matching artifacts, not answer errors. Zero answerable tasks
  lack a gold document and zero gold references dangle. Independent
  third-party human review remains a roadmap item.

## Known limitations

- Questions are synthetic (LLM-written), human-spot-checked rather than fully
  human-authored.
- Pooled (non-exhaustive) relevance judgments; see above.
- OCR noise in degraded documents is uncorrected by design: it is part of
  the haystack, never part of the answer key.
- The alias index driving aggregation/timeline bounding is heuristic;
  entities with unusual name forms may be under-covered.
- `false_premise` saturates on the headline (all systems refuse), so it does not
  drive the sort key; its informative signal is the premise-identification
  diagnostic, which separates the agentic systems (1.0) from the one-shot and
  no-context baselines (0.0). Absence is verified against the pooled and
  entity-complete document set, not proven for the wider release.

## Ethics

All source documents are public records. Tasks are generated exclusively
from already-public text; no new personal information is synthesized or
inferred. Retracted or erroneous tasks are removed in point releases
(`v1.x`), never silently edited.
