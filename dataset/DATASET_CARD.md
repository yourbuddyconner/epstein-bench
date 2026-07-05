# Epstein Bench Dataset Card

**Current release:** none yet — `v1.0` is pending its first generation run.
This card documents the methodology; the *Release statistics* section is
filled in by each release and must be updated whenever a version ships.

## Source corpus

- `aurora2424/Epstein-Files` (Hugging Face): the full public Epstein Files
  release — ~4.1M rows (340GB including raw media bytes). Only the text
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
| `unanswerable` | none (refusal expected) | refusal accuracy |

Generation is fact-first: atomic facts are extracted from clean documents and
questions are written against the fact, in investigator phrasing. Aggregation
questions are **bounded** — scoped to an entity whose candidate documents are
enumerable via an alias index — because unbounded "list all X" gold sets
cannot be verified at corpus scale.

## Verification

Every shipped task passed all of:

1. **Standalone** — interpretable without the source document (concrete
   entities, no deixis, no boilerplate targets).
2. **Answerability** — an independent prompt, shown the gold documents,
   recovers the reference answer (semantic match + token-F1 floor; ≥80% item
   recovery for aggregation).
3. **Necessity** — closed-book and random-distractor attempts fail; for
   multi-document types, no single gold document suffices.
4. **Adjudication** — a stronger model passes/fails with a category.

Unanswerable tasks run stages 1 and 4, plus a generation-time absence check
(top BM25 hits confirmed non-answering). All rejections are logged with the
failing stage (`build/rejected.jsonl`).

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
| generation + gauntlet stages 1–3 + pool judging | `gpt-4o-mini-2024-07-18` |
| adjudication + pool stability re-check | `gpt-4o-2024-08-06` |
| scoring judge (prompt v1) | `gpt-4o-mini-2024-07-18` |

Changing the scoring judge or its prompt is a new benchmark version; scores
across versions are not comparable.

## Release statistics (to fill per release)

- v1.0: task counts by type/split, gauntlet pass rates by stage, pool sizes,
  human spot-check sample size and observed error rate. *Pending first
  generation run.*

## Known limitations

- Questions are synthetic (LLM-written), human-spot-checked rather than fully
  human-authored.
- Pooled (non-exhaustive) relevance judgments; see above.
- OCR noise in degraded documents is uncorrected by design — it is part of
  the haystack, never part of the answer key.
- The alias index driving aggregation/timeline bounding is heuristic;
  entities with unusual name forms may be under-covered.

## Ethics

All source documents are public records. Tasks are generated exclusively
from already-public text; no new personal information is synthesized or
inferred. Retracted or erroneous tasks are removed in point releases
(`v1.x`), never silently edited.
