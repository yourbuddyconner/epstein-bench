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
| scoring judge (prompt v1) | `gpt-4o-mini-2024-07-18` |

Changing the scoring judge or its prompt is a new benchmark version; scores
across versions are not comparable.

## Release statistics

### v1.0 (2026-07-07)

- **Corpus:** entity-complete selection over the full source dataset, a wide
  scan of all 634 parquet shards (~1.38M text-bearing docs) indexed entity
  mentions; 40 notable target people were chosen (LLM notability check) and the
  retrieval corpus is every document mentioning a target plus a 30,000-doc
  random backbone, for a corpus of 83,810 documents and 159,564 chunks.
- **Tasks:** 1,000 (`full`): 823 single_hop / 111 aggregation / 27 timeline /
  7 dossier / 32 unanswerable. `dev`: 44. Multi-document types (timeline,
  dossier) survive verification at low rates, so they remain a small share.
- **Gauntlet:** 1,098 of 4,034 candidates passed (27%). Rejections:
  1,528 standalone, 1,080 answerability, 786 adjudication, 229 necessity.
- **Pooling:** 1,018 of 1,098 kept; 74 dropped as unstable under strong-model
  re-judging, 6 as source-not-supportive. One non-person dossier target was
  retracted, leaving 1,000 final tasks.
- **Reference baselines** (cited / uncited correctness): bm25 0.390 / 0.398,
  hybrid 0.381 / 0.412, dense 0.364 / 0.394, closed_book 0.194 / 0.201,
  parametric 0.175 / 0.220. closed_book and parametric score 0.000 on every
  retrieval-requiring type, evidencing retrieval necessity. **Parametric
  knowledge probe:** single_hop uncited 0.057 (vs closed_book 0.021):
  gpt-4o-mini answers ~5.7% of single-hop facts from training weights alone, a
  small but measurable contamination signal.
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

## Ethics

All source documents are public records. Tasks are generated exclusively
from already-public text; no new personal information is synthesized or
inferred. Retracted or erroneous tasks are removed in point releases
(`v1.x`), never silently edited.
