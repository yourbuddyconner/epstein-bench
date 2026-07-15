# Methodology

## Abstract

Retrieval-augmented generation (RAG) is now the standard way to ground large
language models in a document collection. Most public benchmarks, however,
evaluate it over clean, well-formed passages that bear little resemblance to
production corpora. Epstein Bench evaluates RAG on a real, adversarial corpus:
the full public Epstein Files, a set of court- and Congress-released records
dominated by OCR noise, near-duplicate email threads, and legal boilerplate.

The benchmark targets five competencies drawn from the RAG evaluation
literature[^1][^2]: noise robustness, faithful attribution, information
integration, negative rejection, and false-premise rejection. To these it adds
an attribution-gated correctness metric, claim-level citation precision,
bootstrap confidence intervals, a training-contamination probe, and per-task
token and dollar telemetry for agentic systems. Release v1.0 comprises 1,038
verified questions over a curated 83,810-document corpus, scored by a pinned
strong-model judge.

Seven reference systems — three one-shot retrievers, two no-retrieval controls,
and two multi-turn LLM tool-use agents — yield five findings. Among one-shot
retrievers, lexical and hybrid tie while dense retrieval trails on degraded OCR.
A no-retrieval control scores 0.000 on every retrieval-dependent type,
establishing that the tasks are not solvable from parametric knowledge alone.
Multi-document dossier reconstruction stays near the floor for all systems.
False-premise rejection is universal, but only the agents *diagnose* the
fabrication, naming it in every task they refuse (premise-identification 1.0,
against 0.0 for every non-agent baseline). Finally, the cheaper Sonnet-5 agent
tops the board (0.538 against 0.490 for the best retriever) while the costlier
Opus-4.8 agent merely ties one-shot retrieval: capability tier does not buy
accuracy here, and the agents spend $0.053 and $0.114 per task against a single
model call for each retriever.

## 1. Motivation

Recent RAG benchmarks have formalized the abilities a retrieval-augmented system
must exhibit. RGB[^1] isolates noise robustness, negative
rejection, information integration, and counterfactual robustness. RAGBench[^2]
decomposes generation quality into relevance, utilization,
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

**Source.** The corpus derives from `aurora2424/Epstein-Files` on Hugging Face,[^3]
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
in the corpus, then fabricates a specific interaction — a meeting, call, or
deal — with a prominent public figure those documents never connect them to. The
outside figure is rotated across a diverse pool, so the type does not collapse
onto one name. Anchoring on an entity-complete target is what makes the negative
claim ("no document supports this") bounded and checkable; anchoring elsewhere
would make it unfalsifiable. The premise is always a fabricated *relationship*,
never a perturbed date or place of a real meeting, so a system that finds the
real record cannot be unfairly penalized.

## 5. The evaluation contract

A system under test never imports benchmark code. It consumes `questions.jsonl`
and emits `predictions.jsonl`.

**questions.jsonl**, one task per line:

| field | type | meaning |
|---|---|---|
| `task_id` | str | opaque id, echoed back |
| `type` | str | `single_hop` \| `aggregation` \| `timeline` \| `dossier` \| `unanswerable` \| `false_premise` |
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

Relevance labels follow the TREC pooling methodology.[^4] For each task, the union
of the top-20 results from three diverse retrievers (BM25,[^5] dense embeddings, and
a reciprocal-rank-fusion hybrid[^6]), together with the source documents, forms a
pool; a judge labels each pooled document supports / partial / irrelevant, and
the gold set is the `supports` subset. A sampled re-judgment by the stronger
model drops tasks whose labels are unstable.

Pooled relevance is not exhaustive: a document outside the pool that happens to
state the answer is scored as non-gold. Pool composition is versioned with the
release. In v1.0, 1,018 of 1,098 verified tasks survived pooling. Retracting one
non-person dossier target removed a further 18 tasks, leaving 1,000, to which 38
verified `false_premise` tasks were added for the current release (1,038 total).
The rejection types (`unanswerable`, `false_premise`) carry an empty gold set by
construction and bypass pooling.

## 8. Metrics

**Cited answer correctness (headline).** A prediction is correct only when both
conditions hold: a pinned LLM judge (fixed model and published prompt) rules the
answer equivalent to the reference, and at least one cited document lies in the
pooled gold set. "Support" here is *membership in the pooled gold set* — a
document a judge ruled sufficient to answer from — not a fresh entailment check
of the answer against the cited document's text; a system that answers from
memory and cites a pooled gold document passes the gate (the no-retrieval
controls in §9 bound how much that is worth in practice). This is an
attribution-gated metric in the spirit of RAGBench adherence[^2]: fluency without
grounding earns nothing. Per type:

- `single_hop`, `timeline`: binary cited correctness.
- `aggregation`, `dossier`: item-level F1, where an item counts only if matched
  *and* one of the gated citations is among that item's own recorded supporting
  documents (falling back to the task's gold set only for items without
  recorded supports). A matched item with no supporting citation is an
  unsupported claim: it costs cited precision as well as recall.
- `unanswerable`, `false_premise`: rejection accuracy; a confident answer that
  accepts the (absent or fabricated) premise is a hallucination and scores zero.

Only the first three cited documents count toward the correctness gate, so a
system cannot dump its full retrieval list into `citations` to fish for a chance
gold hit.

The overall score is the unweighted macro-average across types, and it is the
leaderboard sort key. Two reporting decisions surround it. First, we report it
with a **bootstrap 95% confidence interval**[^7], resampled within each type over
1,000 draws. Type counts here are badly skewed — single_hop holds 823 tasks and
dossier holds 7 — so a bare point estimate would hide the fact that two systems
a few hundredths apart are a statistical tie. Second, we report the
task-weighted **micro** average alongside the macro. The macro gives the small
rejection types outsized weight, and the two numbers diverge sharply for
no-retrieval systems (§9).

**Citation precision and recall (diagnostic).** In the spirit of ALCE[^8] (but
against the pooled gold set rather than passage-level entailment), we report,
over answerable tasks, the fraction of a system's cited documents that are gold
(precision) and the fraction of gold documents it cited (recall). Precision is
computed over *everything* the system cited — the list is not truncated — so
citation stuffing pays for itself here even though the correctness gate only
reads the first three. Correctness gates on grounding; these quantify its
quality.

**Premise-identification rate (diagnostic).** For `false_premise`, of the tasks
a system refused, the fraction in which it named the specific fabricated fact
rather than declining in general. This separates grounded refusal from actually
catching the falsehood.

**Uncited correctness (diagnostic).** The same judgments with the citation gate
removed. For retrieval systems it isolates grounding failures (right answer,
wrong or missing citation). For a no-context system it estimates parametric
knowledge of the corpus, that is, training contamination.

**Retrieval diagnostics.** recall@5, recall@20, and nDCG@10[^9] of the `retrieved`
list against the pooled gold set, on answerable tasks. Reported as secondary
columns.

**Operational telemetry (agentic systems).** A system may report per-task token
usage and dollar cost alongside its predictions; the scorer aggregates them into
`tokens_per_task` and `cost_usd_per_task`. This makes the accuracy/cost tradeoff
first-class — an agent that spends two orders of magnitude more tokens for a few
points of accuracy is visible as such rather than flattered by an accuracy-only
ranking. The cheap one-shot baselines report nothing here (a single model call);
the columns read `n/a` for them. Unlike every accuracy metric, telemetry is
**self-reported and unverified** — the scorer cannot audit a submitter's token
counts — so read the cost columns as claims, not measurements.

The judge model and prompt are part of the release. Release v1.0 pins the
scoring judge to a strong model (`gpt-5.5-2026-04-23`) and judge prompt **v2**
(v2 fences the judged answer as untrusted data, so a prediction that embeds
instructions to the judge cannot steer its verdict, and tightens the
aggregation gate to per-item attribution); the judge's correctness agreement
approaches human agreement on this task, consistent with the LLM-as-judge
literature[^10]. The generation and gauntlet stages keep a cheaper model, since
filtering tolerates one. Changing the judge or its prompt constitutes a new
scoring version, and scores are not comparable across judge versions — all
reference scores below were computed under v2.

## 9. Baselines and findings

Seven reference systems, scored through the same submission pipeline used for
external entries.

| system | cited | 95% CI | micro | single_hop | aggregation | timeline | dossier | unanswerable | false_premise | recall@5 | recall@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic (sonnet-5) | 0.538 | [0.50, 0.57] | 0.532 | 0.535 | 0.286 | 0.407 | 0.000 | 1.000 | 1.000 | 0.250 | 0.321 |
| hybrid | 0.490 | [0.45, 0.53] | 0.488 | 0.486 | 0.313 | 0.296 | 0.032 | 0.812 | 1.000 | 0.272 | 0.594 |
| agentic (opus-4-8) | 0.488 | [0.46, 0.52] | 0.460 | 0.463 | 0.170 | 0.296 | 0.000 | 1.000 | 1.000 | 0.272 | 0.357 |
| bm25 | 0.483 | [0.45, 0.52] | 0.475 | 0.476 | 0.247 | 0.333 | 0.000 | 0.844 | 1.000 | 0.271 | 0.589 |
| dense | 0.434 | [0.40, 0.47] | 0.393 | 0.383 | 0.217 | 0.185 | 0.000 | 0.844 | 0.974 | 0.219 | 0.517 |
| closed_book | 0.328 | [0.32, 0.33] | 0.066 | 0.000 | 0.000 | 0.000 | 0.000 | 0.969 | 1.000 | 0.000 | 0.000 |
| parametric | 0.304 | [0.28, 0.32] | 0.062 | 0.000 | 0.000 | 0.000 | 0.000 | 0.875 | 0.947 | 0.000 | 0.000 |

The `agentic` rows are a multi-turn LLM tool-use agent (Claude Sonnet 5 and
Claude Opus 4.8) that issues its own retrieval queries over the hybrid retriever
and then commits an answer; they report per-task token usage and dollar cost
(below). Five findings follow.

1. **Lexical and hybrid retrieval tie; dense trails.** Hybrid (0.490, CI
   [0.45, 0.53]) and BM25 (0.483, [0.45, 0.52]) are statistically
   indistinguishable; their intervals almost entirely overlap. Dense (0.434,
   [0.40, 0.47]) sits clearly below both, its interval barely reaching their
   point estimates. The confidence intervals are what license reading this as a
   tie at the top with dense behind, rather than a spurious three-way rank
   order. The likely mechanism is that BM25 scores a passage on whichever rare
   terms survive: a name or account number that OCR left intact still matches
   exactly, and corruption elsewhere in the passage costs nothing. A dense
   encoder instead pools the whole garbled passage into a single vector, so
   noise anywhere degrades the match. RRF hybridization matches BM25 but does
   not clearly beat it. The ordering inverts the usual clean-benchmark result,
   where dense leads.
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
4. **False-premise rejection is universal; premise *diagnosis* separates the
   agent from everything else.** Every system refuses fabricated-premise
   questions at 0.95–1.00, so the headline `false_premise` score saturates and
   does not discriminate. The `premise_id_rate` diagnostic does: the agentic
   system names the specific fabrication in all 38 tasks (1.000), while every
   retrieval and no-context baseline scores 0.000 — they decline for lack of
   support without articulating *what* is false. This is the axis the type was
   built to expose: a system that reasons over evidence catches the lie; one
   that only fails to ground it does not. The lesson for the metric is that
   `false_premise` belongs in the diagnostics, read through `premise_id_rate`,
   and its saturated headline should not drive the sort key.
5. **An agent tops the board, but capability tier does not buy accuracy.** The
   Sonnet-5 agent (0.538, CI [0.50, 0.57]) leads the best one-shot retriever
   (hybrid 0.490). Issuing several targeted queries and reasoning over the
   results converts evidence into correct, cited answers more effectively. It
   does so even though the agent's recall@20 (0.321) is *lower* than hybrid's
   (0.594): the agent does not need the gold document sitting in a top-20 list,
   only to find it and use it. The more expensive Opus-4.8 agent scores
   *lower* (0.488, a statistical tie with plain hybrid) at more than double the
   cost. Its more conservative, literal tool use depresses aggregation (0.170
   against 0.286) and citation precision (0.368 against 0.485). Cost is
   first-class on the leaderboard: Sonnet spends **$0.053 and ~14,800 tokens per
   task** (~$55 for the split), Opus **$0.114 and ~19,600 tokens** (~$119). Each
   retrieval baseline instead makes a single model call and reports no
   telemetry, so the exact multiple is unmeasured — but it is large, and the
   Opus agent buys nothing over one-shot hybrid for it. The best system is the
   *cheaper* agent, which is exactly the tradeoff the operational columns exist
   to expose.

**Contamination probe.** The `parametric` control is prompted to answer from its
own weights. Its single-hop uncited score is 0.045, against 0.017 for the
abstention-biased `closed_book` control: the generation-time model already
reproduces roughly 4–5% of single-hop facts without retrieval. Because the
benchmark is decontaminated only against that generation-time model (the
necessity stage rejects tasks it can answer closed-book), any *other* model's
parametric score is a clean measure of its own training exposure to the public
release, and this quantity is expected to rise for models trained after the
files entered the public web.

## 10. Reproducibility and submissions

The pipeline (`scan → select → generate → verify → pool → finalize`) is seeded
and config-pinned, and all LLM calls are disk-cached, so an interrupted run
resumes without re-spending. Two honest caveats on "reproducible": the
generator models are not bit-deterministic, so a cold re-run (empty cache)
produces a *comparable* dataset, not an identical one — byte-identical
regeneration relies on the released LLM cache; and the shipped v1.0 split
includes two manual steps the pipeline does not encode (one non-person dossier
target retracted after pooling, and the 38 verified `false_premise` tasks
appended, §7). The released split is pinned by the hashes in
`dataset/v1.0/manifest.json`, which validation enforces.

A submission is built with `python -m epstein_bench submit`, which packages
predictions, system metadata, and the dataset-version hash. A submission PR
also commits its `scores.json`; a GitHub Action **rescores the raw predictions
with the base branch's scorer and dataset and fails unless the committed
scores match the recomputation** (within a small tolerance for judge
nondeterminism), so the leaderboard never rests on self-reported numbers. The
validation run takes only the `submissions/` directory from the PR — dataset or
scorer edits in the same PR have no effect on its own validation. Because this
is a self-run benchmark, the full answer key (`tasks.jsonl`) ships next to the
questions; recomputation proves the scores follow from the predictions, but it
cannot prove the predictions were produced without reading the key. Leaderboard
integrity therefore ultimately rests on submission review, as in every
open-answer-key benchmark. The `dev` split is for iteration and is not
leaderboard-eligible.

## 11. Limitations

- Questions are LLM-generated and machine-verified, with author spot-checking,
  rather than fully human-authored. Independent third-party human review is a
  roadmap item.
- One cheap model (the generation-time model) both drafts tasks and runs the
  answerability/necessity stages of the gauntlet, so the released tasks are
  biased toward questions that model family finds answerable, and a consistent
  misreading of OCR by that model can survive verification. The strong model
  enters only at adjudication and pool-stability checks.
- Pooled relevance is non-exhaustive (§7) — and because the headline metric is
  attribution-gated, this affects *correctness*, not just retrieval recall: a
  right answer citing a genuinely supporting document outside the pool scores
  zero. Systems whose retrieval diverges from the three pooling retrievers are
  structurally disadvantaged; a doc-addition appeals process is a roadmap item.
- Pool and gauntlet judges read a fixed-length excerpt of each document (2,500
  characters in v1.0; 3,000 in the current pipeline), while systems under test
  see full documents. A supporting passage past the cutoff can leave a genuinely
  relevant document out of the gold set.
- The generation-time absence check for `unanswerable` and `false_premise`
  verifies absence against the top lexical (BM25) hits for the question, with
  strong-model adjudication over those same documents — not against the full
  pooled or entity-complete document set. A premise supported only by a document
  those probes miss could slip through.
- OCR noise is uncorrected by design; it is part of the haystack but never part
  of the answer key.
- The alias index that bounds aggregation and dossier tasks is heuristic;
  entities with unusual name forms may be under-covered.
- The `false_premise` headline saturates (every system refuses), so it does not
  separate systems on the sort key; its informative signal is the
  premise-identification diagnostic, which separates the agentic system (1.0)
  from the one-shot and no-context baselines (0.0).
- The `agentic` references are nondeterministic and depend on an external
  provider (an Anthropic key), so their exact numbers are not reproducible to
  the digit and will drift with the model; they are reference points, not fixed
  controls like the retrieval baselines. Their token/cost figures are
  sticker-price estimates, and telemetry in general is self-reported (§8).
- Decontamination holds only against the generation-time model. Contamination
  scores for other models are informative but not a guarantee of task novelty
  for those models.
- Correctness depends on a pinned LLM judge; a different judge is a different
  benchmark version.

## Notes

[^1]: Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented
  Generation" (RGB), 2023. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)

[^2]: Friel, Belyi, and Sanyal, "RAGBench: Explainable Benchmark for
  Retrieval-Augmented Generation Systems," 2024.
  [arXiv:2407.11005](https://arxiv.org/abs/2407.11005)

[^3]: The public release, mirrored on Hugging Face:
  [huggingface.co/datasets/aurora2424/Epstein-Files](https://huggingface.co/datasets/aurora2424/Epstein-Files)

[^4]: Voorhees, "The Philosophy of Information Retrieval Evaluation," CLEF 2001,
  describes the TREC pooling methodology for incomplete relevance judgments.
  [doi:10.1007/3-540-45691-0_34](https://link.springer.com/chapter/10.1007/3-540-45691-0_34)

[^5]: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and
  Beyond," Foundations and Trends in Information Retrieval, 2009.
  [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019)

[^6]: Cormack, Clarke, and Buettcher, "Reciprocal Rank Fusion Outperforms
  Condorcet and Individual Rank Learning Methods," SIGIR 2009.
  [doi:10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)

[^7]: Efron and Tibshirani, *An Introduction to the Bootstrap*, Chapman and
  Hall, 1993. [doi:10.1201/9780429246593](https://doi.org/10.1201/9780429246593)

[^8]: Gao et al., "Enabling Large Language Models to Generate Text with
  Citations" (ALCE), EMNLP 2023.
  [arXiv:2305.14627](https://arxiv.org/abs/2305.14627)

[^9]: Järvelin and Kekäläinen, "Cumulated Gain-Based Evaluation of IR
  Techniques," ACM TOIS, 2002.
  [doi:10.1145/582415.582418](https://doi.org/10.1145/582415.582418)

[^10]: Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,"
  NeurIPS 2023. [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
