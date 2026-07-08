# Scoring hardening + `false_premise` task family — design

**Date:** 2026-07-07
**Status:** approved (pre-release, no version bump)

Four changes to Epstein Bench, motivated by a review against the RAG-eval
frontier (ALCE citation precision, RAGChecker component separation, ARES
statistical confidence, CRAG/RGB false-premise & counterfactual rejection).
All land in the current pre-release; no dataset version increment.

## 1. Bootstrap confidence intervals on scores

**Problem.** The headline is an unweighted macro-average over five types whose
counts are wildly unequal (823 / 111 / 27 / 7 / 32). Dossier (n=7) contributes
20% of the headline and a single task swings it ~2.8 points. No uncertainty is
reported, so leaderboard entries separated by <0.02 are indistinguishable from
noise.

**Change.** In `score.py`, retain per-task `(type, cited, uncited)` tuples and
compute a bootstrap 95% CI (resample tasks with replacement, recompute the macro
across types each iteration) for `overall_cited_correctness` and
`overall_uncited_correctness`, plus per-type CIs. Deterministic: seed the RNG
from `config.seed`. Report as `overall_cited_correctness_ci95: [lo, hi]` etc.
Default 1000 iterations, configurable via `config.bootstrap_iterations`.

Micro (task-weighted) average reported alongside macro as
`overall_cited_correctness_micro` so readers can see both.

## 2. Citation precision + close the "cite everything" gate

**Problem.** For `single_hop`/`timeline`, credit is granted when
`any(c in gold for c in citations)` and citations are capped only at
`max_retrieved` (20). A system can cite its whole top-20 list, land one gold doc
by luck, and collect full credit — no penalty for 19 unsupported citations. This
tests citation *recall* only, never precision (cf. ALCE).

**Change.**
- Add `config.gate_max_citations = 3`: only the first N citations count toward
  the correctness gate. Committing to a small supporting set is now required;
  dumping the retrieval list no longer games the gate. Documented in the
  contract.
- Report `citation_precision` (fraction of a prediction's cited docs that are in
  the pooled gold set) and `citation_recall` (fraction of gold docs cited),
  averaged over answered, answerable tasks, as diagnostics. For
  aggregation/dossier, computed against the union of per-item gold + pooled gold.

The headline metric semantics stay binary (correct + supported) — only the gate
input is capped and the diagnostics are added. No dataset regen required.

## 3. Stronger scoring judge (gpt-5.5)

**Problem.** The correctness judge is `gpt-4o-mini`, uncalibrated against humans
— the most attackable design choice in the paper.

**Change.** `config.judge_model = "gpt-5.5-2026-04-23"` (pinned snapshot).
Generation and gauntlet stages 1–3 keep the cheap model (initial filtering is
fine on a small model); adjudication keeps `strong_model`. Only the scoring
judge and the aggregation judge (both already read `config.judge_model`) move.

**Compatibility.** The GPT-5 family rejects `temperature != 1` and `seed` on
chat.completions (verified: 400 "temperature does not support 0"). `_openai_chat`
must omit `temperature` and `seed` for models matching the GPT-5 family; the
bare call (json response_format only) succeeds. Determinism for the judge is then
provided by snapshot pinning + the disk cache, not by seed.

**Consequence.** Every published score changes. Re-run scoring on the saved
baseline predictions (`build/preds_*.jsonl` / `submissions/*/predictions.jsonl`)
and refresh `docs/leaderboard.json`, the methodology baselines table, and the
dataset card. Pre-release, so this is the right time.

## 4. `false_premise` task family

**Concept.** Questions that presuppose a specific, fabricated relationship or
event between real, in-corpus entities. Target behavior: reject the premise
rather than answer the follow-up. Distinct from `unanswerable` (which is about
entities/facts *absent* from the corpus): `false_premise` is about entities that
are heavily *present*, with an invented connection. The failure mode tested is
sycophantic acceptance of a false presupposition — on-theme for a corpus where
inventing connections between real people is the exact harm to avoid.

**Anchoring (load-bearing constraint).** Premises are anchored only on the ~40
entity-complete target people. The corpus holds *every* document mentioning a
target, so "no document supports proposition P" is a bounded, checkable claim for
those entities — the same property that makes dossiers verifiable, reused for
negation. Anchoring elsewhere would make absence unfalsifiable.

**(a) Generation.** Fact-first, inverted. Take a real salient fact about a
target (a real counterparty / place / date already extracted), perturb exactly
one element into a plausible falsehood (swap counterparty for another real
corpus figure; move a documented meeting to a place they never co-occur; invent
a transaction). Perturbing a real fact keeps the premise plausible and
standalone. Prompt tag `[FALSEPREMISE]`. Record the perturbed element as
`false_element` for scoring the identification diagnostic.

**(b) Verification gauntlet.**
1. **Standalone** (reuse stage 1): concrete entities, no deixis.
2. **Absence** (new, the crux): pool top-K over BM25 *and* dense using premise
   terms + the target's alias set; a judge reads pooled candidates and answers
   "does any document support proposition P?" Entity-completeness makes a clean
   "no support" trustworthy. Any support → discard.
3. **Adjudication** (reuse stage 4, strong model): premise is (i) plausible,
   (ii) actually false given the entity's docs, (iii) not accidentally
   answerable.

**(c) Scoring.** Headline = refusal accuracy, reusing the `unanswerable` scoring
path exactly (pass = declines / flags the premise unsupported); folds into the
negative-rejection macro and needs no new judge calibration for the sort key.
Diagnostic = `premise_id_rate`: of refused tasks, the fraction where the system
correctly named which presupposed fact is false (LLM-judged against
`false_element`). Reported, not in the headline.

**Scope.** ~30 shipped tasks, sized like `unanswerable`; added to
`config.type_mix`. Reuses stages 1 & 4 and pooling. New code: the
`[FALSEPREMISE]` generator and the absence stage. Requires re-running
`generate → verify` for this type (API spend).

## Non-goals

- No dataset version bump (pre-release).
- No change to corpus construction, entity selection, or pooling mechanics.
- No span/chunk-level retrieval gold (separate roadmap item).
- No multi-judge panel (single stronger judge for now).
