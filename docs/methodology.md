# Epstein Bench: Methodology

## Abstract

Retrieval-augmented generation (RAG) is now the standard way to ground large
language models in a document collection, but most public benchmarks evaluate it
over clean, well-formed passages that bear little resemblance to production
corpora. Epstein Bench evaluates RAG on a real, adversarial corpus: the full
public Epstein Files, a set of court- and Congress-released records dominated by
OCR noise, near-duplicate email threads, and legal boilerplate. The benchmark
targets four competencies drawn from the RAG evaluation literature (noise
robustness, faithful attribution, information integration, and negative
rejection) and adds an attribution-gated correctness metric together with a
training-contamination probe. Release v1.0 comprises 1,000 verified questions
over a curated 83,810-document corpus. Reference systems reveal three findings:
lexical retrieval (BM25) outperforms dense retrieval on heavily degraded text; a
no-retrieval control scores 0.000 on every retrieval-dependent task type,
establishing that the tasks are not solvable from parametric knowledge alone;
and multi-document reconstruction (person timelines) remains near the floor for
all baselines.

## 1. Motivation

Recent RAG benchmarks have formalized the abilities a retrieval-augmented system
must exhibit. RGB (arXiv:2309.01431) isolates noise robustness, negative
rejection, information integration, and counterfactual robustness. RAGBench
(arXiv:2407.11005) decomposes generation quality into relevance, utilization,
adherence (faithfulness), and completeness. Both construct their evaluation data
from curated or synthetically clean sources: contemporary news, well-formed
domain passages.

Production retrieval rarely operates on clean text. Enterprise and investigative
corpora arrive as scanned PDFs, inconsistent OCR, forwarded email chains, and
duplicated attachments, with the answer to any given question buried in a small
number of documents among hundreds of thousands. A benchmark that never presents
this degradation cannot predict how a system will behave when it does.

Epstein Bench holds the abilities fixed and changes the corpus. It measures the
same competencies the literature has established, on a corpus whose noise is
real rather than injected, and it scores answers only when they are attributable
to a cited source document.

## 2. Abilities measured

Each competency maps to a concrete surface of the benchmark.

| Ability (literature) | Realized in Epstein Bench as |
|---|---|
| Noise robustness | The corpus is uncorrected OCR of scanned emails, depositions, and financial records. Retrieval and reading must tolerate genuine document degradation, not synthetic perturbation. |
| Faithful attribution | Scoring is attribution-gated: an answer earns credit only if it is correct *and* a cited document supports it (§8). |
| Information integration | The `aggregation`, `timeline`, and `dossier` task types require synthesizing evidence across multiple documents; single-document context is provably insufficient (§6). |
| Negative rejection | The `unanswerable` type poses plausible questions whose answers are absent from the corpus; the target behavior is abstention. |

## 3. Corpus construction

**Source.** The corpus derives from `aurora2424/Epstein-Files` on Hugging Face,
roughly 4.1M released records (about 340 GB including media). Only text-bearing
rows are used, read directly from the parquet shards with column projection to
avoid downloading media bytes. A full scan of all 634 shards yields
approximately 1.38M documents with extractable text.

**Quality screening.** Each document is classified `clean`, `degraded`, or
`garbage` by a heuristic pass (character-level garbage ratio, dictionary-word
ratio, length) with a cheap-model readability check on borderline cases. Tasks
are generated only from `clean` text. `degraded` documents remain in the
retrieval corpus as natural distractors. `garbage` is excluded.

**Entity-complete selection.** Constructing honest multi-document tasks about a
person requires that the corpus contain *all* of that person's documents;
otherwise the gold evidence set is silently incomplete. We therefore index
entity mentions across the full scan, select 40 notable target entities via a
notability classifier (public figures only; entities exceeding a document-count
ceiling are excluded as too pervasive to bound), and assemble the retrieval
corpus from every document mentioning a target plus a seeded random backbone of
30,000 additional documents. Release v1.0 contains 83,810 documents segmented
into 159,564 retrieval chunks.

## 4. Task taxonomy

Generation is fact-first: atomic, verifiable facts are extracted from clean
documents, and each question is written against a fact in investigator phrasing.
Single-hop facts are additionally salience-filtered (retained only above a
newsworthiness threshold, and only when document-stated) so the benchmark
reflects the substance of the corpus rather than administrative trivia.

| Type | Definition | Gold | Ability |
|---|---|---|---|
| `single_hop` | Factoid answerable from one document | Short answer + supporting docs | Noise robustness, attribution |
| `aggregation` | Bounded list scoped to an entity | Item set, per-item supporting docs | Information integration |
| `timeline` | Temporal ordering or span over 2+ documents | Answer + supporting docs | Information integration |
| `dossier` | Documented timeline of one person's contacts | Dated event list, per-item docs | Information integration |
| `unanswerable` | Plausible question with no corpus answer | Abstention expected | Negative rejection |

Aggregation is *bounded*: scoped to an entity whose candidate documents are
enumerable through the alias index, because unbounded "list all X" gold sets
cannot be verified at corpus scale. Dossier tasks exploit the entity-complete
selection of §3, which is what makes a claimed person timeline verifiable.

## 5. The evaluation contract

A system under test never imports benchmark code. It consumes `questions.jsonl`
and emits `predictions.jsonl`.

**questions.jsonl**, one task per line:

| field | type | meaning |
|---|---|---|
| `task_id` | str | opaque id, echoed back |
| `type` | str | `single_hop` \| `aggregation` \| `timeline` \| `dossier` \| `unanswerable` |
| `question` | str | the question |

**predictions.jsonl**, one prediction per line, every task answered:

| field | type | meaning |
|---|---|---|
| `task_id` | str | must match a released task |
| `answer` | str | free text; explicit abstention when the corpus lacks the answer |
| `citations` | [str] | document ids the system claims support the answer |
| `retrieved` | [str] | ranked retrieval list, at most 20 ids |

Document ids are the `doc_id` values of the source records. A system is not told
a task's type beyond the field itself; withholding the `type` field from the
pipeline is permitted and closer to production conditions.

## 6. Verification

Every released task passes a four-stage gauntlet; a failure at any stage
discards the task, and the failing stage is logged.

1. **Standalone.** The question is interpretable without the source document:
   concrete entities, no deixis, no boilerplate targets.
2. **Answerability.** An independent prompt, shown the gold documents, recovers
   the reference answer (semantic match under an LLM judge, with a token-overlap
   floor; for list types, an item-recovery threshold).
3. **Necessity.** Closed-book and random-distractor attempts must fail. For
   multi-document types, no single gold document may suffice, which enforces the
   information-integration property.
4. **Adjudication.** A stronger model issues a final pass or fail with a
   category.

Unanswerable tasks run stages 1 and 4 plus a generation-time absence check
against top lexical hits. In v1.0, 1,098 of 4,034 candidates (27%) survived the
gauntlet.

## 7. Retrieval ground truth by pooling

Relevance labels follow the TREC pooling methodology. For each task, the union
of the top-20 results from three diverse retrievers (BM25, dense embeddings, and
reciprocal-rank-fusion hybrid), together with the source documents, forms a
pool; a judge labels each pooled document supports / partial / irrelevant, and
the gold set is the `supports` subset. A sampled re-judgment by the stronger
model drops tasks whose labels are unstable.

Pooled relevance is not exhaustive: a document outside the pool that happens to
state the answer is scored as non-gold. Pool composition is versioned with the
release. In v1.0, 1,018 of 1,098 verified tasks survived pooling; one non-person
dossier target was retracted, leaving 1,000 released tasks.

## 8. Metrics

**Cited answer correctness (headline).** A prediction is correct only when both
conditions hold: a pinned LLM judge (fixed model and published prompt) rules the
answer equivalent to the reference, and at least one cited document lies in the
pooled gold set. This is an attribution-gated metric in the sense of RAGBench
adherence: fluency without grounding earns nothing. Per type:

- `single_hop`, `timeline`: binary cited correctness.
- `aggregation`, `dossier`: item-level F1, where an item counts only if matched
  *and* supported by a cited document.
- `unanswerable`: abstention accuracy; a confident wrong answer is a
  hallucination and scores zero.

The overall score is the unweighted macro-average across types and is the
leaderboard sort key.

**Uncited correctness (diagnostic).** The same judgments with the citation gate
removed. For retrieval systems it isolates grounding failures (right answer,
wrong or missing citation). For a no-context system it estimates parametric
knowledge of the corpus, that is, training contamination.

**Retrieval diagnostics.** recall@5, recall@20, and nDCG@10 of the `retrieved`
list against the pooled gold set, on answerable tasks. Reported as secondary
columns.

The judge model and prompt are part of the release. Changing either constitutes
a new benchmark version, and scores are not comparable across judge versions.

## 9. Baselines and findings

Five reference systems, scored through the same submission pipeline used for
external entries.

| system | cited | uncited | single_hop | aggregation | timeline | dossier | unanswerable | recall@5 | recall@20 |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.390 | 0.398 | 0.471 | 0.271 | 0.296 | 0.036 | 0.875 | 0.271 | 0.589 |
| hybrid | 0.381 | 0.412 | 0.481 | 0.326 | 0.222 | 0.032 | 0.844 | 0.272 | 0.594 |
| dense | 0.364 | 0.394 | 0.377 | 0.249 | 0.185 | 0.071 | 0.938 | 0.219 | 0.517 |
| closed_book | 0.194 | 0.201 | 0.000 | 0.000 | 0.000 | 0.000 | 0.969 | 0.000 | 0.000 |
| parametric | 0.175 | 0.220 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 |

Three findings follow.

1. **Lexical retrieval beats dense on degraded text.** BM25 (0.390) exceeds
   dense retrieval (0.364), and RRF hybridization does not recover the gap
   (0.381). On corrupted OCR, sub-word lexical matching is more robust than
   dense-vector similarity over embeddings of garbled tokens. This is a direct
   noise-robustness result and inverts the usual ordering on clean benchmarks.
2. **The tasks require retrieval.** The `closed_book` and `parametric` controls,
   which receive no documents, score 0.000 on every retrieval-dependent type.
   Their nonzero overall scores are entirely abstention accuracy on the
   `unanswerable` type. The tasks cannot be solved from parametric priors.
3. **Information integration is unsolved.** Dossier reconstruction, which
   requires assembling a person's timeline across many documents, stays between
   0.03 and 0.07 for every system. Multi-document synthesis on a noisy corpus is
   the open frontier this benchmark exposes.

**Contamination probe.** The `parametric` control is prompted to answer from its
own weights. Its single-hop uncited score is 0.057, against 0.021 for the
abstention-biased `closed_book` control: the generation-time model already
reproduces roughly 5.7% of single-hop facts without retrieval. Because the
benchmark is decontaminated only against that generation-time model (the
necessity stage rejects tasks it can answer closed-book), any *other* model's
parametric score is a clean measure of its own training exposure to the public
release, and this quantity is expected to rise for models trained after the
files entered the public web.

## 10. Reproducibility and submissions

The pipeline (`scan → select → generate → verify → pool → finalize`) is seeded
and config-pinned. All LLM calls are disk-cached, so an interrupted run resumes
without re-spending. A submission is built with
`python -m epstein_bench submit`, which packages predictions, system metadata,
and the dataset-version hash. A GitHub Action validates the bundle and
**recomputes all scores from the raw predictions**; self-reported numbers are
never used. The `dev` split is for iteration and is not leaderboard-eligible.

## 11. Limitations

- Questions are LLM-generated and machine-verified, with author spot-checking,
  rather than fully human-authored. Independent third-party human review is a
  roadmap item.
- Pooled relevance is non-exhaustive (§7).
- OCR noise is uncorrected by design; it is part of the haystack but never part
  of the answer key.
- The alias index that bounds aggregation and dossier tasks is heuristic;
  entities with unusual name forms may be under-covered.
- Decontamination holds only against the generation-time model. Contamination
  scores for other models are informative but not a guarantee of task novelty
  for those models.
- Correctness depends on a pinned LLM judge; a different judge is a different
  benchmark version.

## References

- Benchmarking Large Language Models in Retrieval-Augmented Generation.
  arXiv:2309.01431.
- RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems.
  arXiv:2407.11005.
- The pooling methodology for incomplete relevance judgments follows standard
  TREC practice.
