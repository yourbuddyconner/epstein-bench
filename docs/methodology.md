# Epstein Bench: Methodology

## Abstract

Retrieval-augmented generation (RAG) is now the standard way to ground large
language models in a document collection, but most public benchmarks evaluate it
over clean, well-formed passages that bear little resemblance to production
corpora. Epstein Bench evaluates RAG on a real, adversarial corpus: the full
public Epstein Files, a set of court- and Congress-released records dominated by
OCR noise, near-duplicate email threads, and legal boilerplate. The benchmark
targets five competencies drawn from the RAG evaluation literature (noise
robustness, faithful attribution, information integration, negative rejection,
and false-premise rejection) and adds an attribution-gated correctness metric,
claim-level citation precision, bootstrap confidence intervals, and a
training-contamination probe. Release v1.0 comprises 1,038 verified questions
over a curated 83,810-document corpus, scored by a pinned strong-model judge.
Reference systems reveal four findings: lexical and hybrid retrieval are a
statistical tie at the top while dense retrieval trails on heavily degraded
text; a no-retrieval control scores 0.000 on every retrieval-dependent task
type, establishing that the tasks are not solvable from parametric knowledge
alone; multi-document reconstruction (person timelines) remains near the floor
for all baselines; and every baseline refuses fabricated-premise questions yet
none identifies the specific falsehood, so grounded rejection is solved while
premise diagnosis is open.

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
| False-premise rejection | The `false_premise` type poses questions that presuppose a fabricated interaction between a real, in-corpus person and a prominent outside figure; the target behavior is to reject the premise rather than answer the follow-up (§4). |

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
| `false_premise` | Question presupposing a fabricated interaction between a real person and a prominent outside figure | Rejection expected; `false_element` records the fabrication | False-premise rejection |

Aggregation is *bounded*: scoped to an entity whose candidate documents are
enumerable through the alias index, because unbounded "list all X" gold sets
cannot be verified at corpus scale. Dossier tasks exploit the entity-complete
selection of §3, which is what makes a claimed person timeline verifiable.

False-premise tasks reuse that same entity-complete property for the opposite
purpose. A premise is anchored on a target person whose *entire* document set is
in the corpus, then fabricates a specific interaction (a meeting, call, or deal)
with a prominent public figure — rotated across a diverse pool so the type does
not collapse onto one name — whom those documents never connect them to.
Anchoring on an entity-complete target is what makes the negative claim ("no
document supports this") bounded and checkable; anchoring elsewhere would make
it unfalsifiable. The premise is a fabricated *relationship*, never a perturbed
date or place of a real meeting, so a system that finds the real record cannot
be unfairly penalized.

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

False-premise tasks also skip stages 2–3 (they carry no gold answer to recover)
and instead pass a generation-time absence check plus a two-stage adjudication.
The absence check retrieves the top lexical hits for the presupposing question
and confirms they do not answer it. Adjudication then runs two focused
strong-model judgments rather than one omnibus prompt, which proved unstable on
exactly the cases that matter: a neutral *support* check that drops any premise
the on-topic documents actually support (a premise built by perturbing only a
detail of a real meeting survives the absence check but is caught here), and a
*quality* check that the premise is concrete and plausible and that its wording
does not signal that it is false.

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
dossier target was retracted, leaving 1,000 tasks, to which 38 verified
`false_premise` tasks were added for the current release (1,038 total). The
rejection types (`unanswerable`, `false_premise`) carry an empty gold set by
construction and bypass pooling.

## 8. Metrics

**Cited answer correctness (headline).** A prediction is correct only when both
conditions hold: a pinned LLM judge (fixed model and published prompt) rules the
answer equivalent to the reference, and at least one cited document lies in the
pooled gold set. This is an attribution-gated metric in the sense of RAGBench
adherence: fluency without grounding earns nothing. Per type:

- `single_hop`, `timeline`: binary cited correctness.
- `aggregation`, `dossier`: item-level F1, where an item counts only if matched
  *and* supported by a cited document.
- `unanswerable`, `false_premise`: rejection accuracy; a confident answer that
  accepts the (absent or fabricated) premise is a hallucination and scores zero.

Only the first three cited documents count toward the correctness gate, so a
system cannot dump its full retrieval list into `citations` to fish for a chance
gold hit. The overall score is the unweighted macro-average across types and is
the leaderboard sort key. We report it with a **bootstrap 95% confidence
interval** (resampled within each type, 1,000 draws): with type counts as skewed
as these (single_hop 823, dossier 7), a point estimate alone hides that two
systems a hundredths apart are a statistical tie. We also report the
task-weighted **micro** average alongside the macro, because the macro gives the
small rejection types outsized weight and the two numbers diverge sharply for
no-retrieval systems (§9).

**Citation precision and recall (diagnostic).** Following ALCE, we report, over
answerable tasks, the fraction of a system's cited documents that are gold
(precision) and the fraction of gold documents it cited (recall). Correctness
gates on grounding; these quantify its quality.

**Premise-identification rate (diagnostic).** For `false_premise`, of the tasks
a system refused, the fraction in which it named the specific fabricated fact
rather than declining in general. This separates grounded refusal from actually
catching the falsehood.

**Uncited correctness (diagnostic).** The same judgments with the citation gate
removed. For retrieval systems it isolates grounding failures (right answer,
wrong or missing citation). For a no-context system it estimates parametric
knowledge of the corpus, that is, training contamination.

**Retrieval diagnostics.** recall@5, recall@20, and nDCG@10 of the `retrieved`
list against the pooled gold set, on answerable tasks. Reported as secondary
columns.

The judge model and prompt are part of the release. Release v1.0 pins the
scoring judge to a strong model (`gpt-5.5-2026-04-23`), whose correctness
agreement approaches human on this task; the generation and gauntlet stages keep
a cheaper model, since filtering tolerates one. Changing the judge or its prompt
constitutes a new benchmark version, and scores are not comparable across judge
versions.

## 9. Baselines and findings

Five reference systems, scored through the same submission pipeline used for
external entries.

| system | cited | 95% CI | micro | single_hop | aggregation | timeline | dossier | unanswerable | false_premise | recall@5 | recall@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid | 0.494 | [0.46, 0.54] | 0.489 | 0.484 | 0.341 | 0.296 | 0.032 | 0.812 | 1.000 | 0.272 | 0.594 |
| bm25 | 0.492 | [0.45, 0.53] | 0.482 | 0.478 | 0.297 | 0.333 | 0.000 | 0.844 | 1.000 | 0.271 | 0.589 |
| dense | 0.440 | [0.40, 0.47] | 0.398 | 0.384 | 0.255 | 0.185 | 0.000 | 0.844 | 0.974 | 0.219 | 0.517 |
| closed_book | 0.328 | [0.32, 0.33] | 0.066 | 0.000 | 0.000 | 0.000 | 0.000 | 0.969 | 1.000 | 0.000 | 0.000 |
| parametric | 0.304 | [0.28, 0.32] | 0.062 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.947 | 0.000 | 0.000 |

Four findings follow.

1. **Lexical and hybrid retrieval tie; dense trails.** Hybrid (0.494, CI
   [0.46, 0.54]) and BM25 (0.492, [0.45, 0.53]) are statistically
   indistinguishable — their intervals almost entirely overlap — while dense
   (0.440, [0.40, 0.47]) sits clearly below both, its interval not reaching
   their point estimates. On corrupted OCR, sub-word lexical matching is more
   robust than dense-vector similarity over embeddings of garbled tokens, and
   RRF hybridization matches but does not clearly beat BM25. The confidence
   intervals are what license reading this as "a tie at the top with dense
   behind" rather than a spurious three-way rank order; the result inverts the
   usual clean-benchmark ordering, where dense leads.
2. **The tasks require retrieval, and the macro can hide it.** The `closed_book`
   and `parametric` controls score 0.000 on every retrieval-dependent type;
   their nonzero overall is entirely rejection accuracy on `unanswerable` and
   `false_premise`. The macro-average flatters them (closed_book 0.328) because
   it weights the two small rejection types equally with the 823-task
   `single_hop` type; the task-weighted micro (closed_book 0.066, parametric
   0.062) exposes the floor the macro conceals. This divergence is the clearest
   argument for reporting both numbers.
3. **Information integration is unsolved.** Dossier reconstruction stays between
   0.000 and 0.032 for every system under the strong judge. Multi-document
   synthesis on a noisy corpus is the open frontier this benchmark exposes.
4. **Grounded rejection is solved; premise diagnosis is not.** Every baseline
   refuses fabricated-premise questions at 0.95–1.00: a retrieval system grounds
   in context that does not support the invented meeting and declines, and a
   no-retrieval system declines for lack of any basis. But the
   premise-identification rate is 0.000 across all five — none names the
   specific fabrication. `false_premise` therefore currently measures grounded
   rejection, which the references already achieve, and exposes premise
   diagnosis as headroom for systems that reason about *why* a premise fails.
   Because the type saturates, it lifts every macro uniformly and does not
   separate the references; its discriminating value is the identification-rate
   diagnostic, not the headline.

**Contamination probe.** The `parametric` control is prompted to answer from its
own weights. Its single-hop uncited score is 0.041, against 0.016 for the
abstention-biased `closed_book` control: the generation-time model already
reproduces roughly 4% of single-hop facts without retrieval. Because the
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
- The `false_premise` type saturates on the reference baselines (all refuse),
  so it does not currently separate retrieval systems on the headline; its
  informative signal is the premise-identification diagnostic, which is 0.000
  for every reference system. Absence is verified against the pooled and
  entity-complete document set, not a proof of non-existence in the wider
  release.
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
