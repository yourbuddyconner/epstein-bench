# Epstein Bench Dataset Card

**Current release:** `v1.0` (2026-07-05). This card documents the
methodology; the *Release statistics* section is updated whenever a version
ships.

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
| `dossier` | dated event list for a notable person, per-item docs | item-level P/R/F1, citation-gated |
| `unanswerable` | none (refusal expected) | refusal accuracy |

Since v1.1, corpus selection is **entity-complete**: a wide scan indexes
entity mentions across the source dataset, an LLM notability check picks
target people (public figures only; entities appearing in more than
`max_entity_docs` documents are excluded as impractically pervasive), and the
corpus is all documents mentioning any target plus a seeded random backbone.
Single-hop facts are additionally **salience-filtered** (newsworthiness ≥3/5:
notable people, money flows, meetings/travel, legal exposure — never
speculation; facts must be document-stated).

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

## Release statistics

### v1.0 (2026-07-05)

- **Corpus:** first 20,000 text-bearing documents of `aurora2424/Epstein-Files`;
  quality screen: 15,637 clean / 823 degraded / 3,540 garbage
  (1,067 borderline docs resolved by LLM readability check); 28,032 chunks;
  5,655-entity alias index.
- **Tasks:** 973 (`full`): 854 single_hop / 47 timeline / 38 aggregation /
  34 unanswerable. `dev`: 88. The mix is single-hop-heavy relative to the
  50/20/15/15 generation quotas because multi-document candidates survive
  verification at much lower rates.
- **Gauntlet:** 1,038 of 5,000 candidates passed (20.8%). Rejections:
  1,528 standalone, 1,448 answerability, 751 adjudication, 234 necessity,
  1 error.
- **Pooling:** 973 of 1,038 kept; 61 dropped as unstable under strong-model
  re-judging, 4 as source-not-supportive.
- **Reference baselines** (overall cited correctness): bm25 0.607,
  hybrid 0.575, dense 0.493, closed_book 0.243 — closed-book scores 0.000
  on every retrieval-requiring type, evidencing retrieval necessity;
  its overall score is entirely refusal accuracy on unanswerable tasks.
- Since v1.1 the scorer also reports **uncited correctness** and baselines
  include a `parametric` mode (answer purely from model weights) — a
  per-model probe of training exposure to the released files. The
  retrieval-necessity control (`closed_book`) is unchanged.
- **Human spot-check:** *pending* — 100-task sample generated
  (`scripts/make_spotcheck.py`, seed 20260705); observed error rate will be
  recorded here and BAD tasks retracted in v1.1.

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
