# Auepora Evaluation Metrics – Engineering Spec

## 1. Purpose and Scope

This document specifies the **full evaluation metrics** to implement for the Auepora-based RAG evaluation library.

Goals:

* Define a **taxonomy of metrics** aligned with Auepora’s target space:

  * Retrieval-level targets
  * Generation-level targets
  * System-level / robustness / additional targets
* For each metric, specify:

  * Target category
  * Required inputs (fields on `EvaluationSample` and `SystemOutputs`)
  * Mathematical definition / algorithm
  * Edge-case behavior
  * Configuration options
* Map metrics to concrete Python modules and classes under `auepora_eval.metrics.*`.

The engineer should treat this as the implementation contract for v1 of the metrics subsystem.

## 2. Metric Taxonomy and Modules

### 2.1 Target Categories (recap)

We reuse the `TargetCategory` enum defined in `core.types`:

* Retrieval:

  * `RETRIEVAL_RELEVANCE`
  * `RETRIEVAL_ACCURACY`
* Generation:

  * `GENERATION_RELEVANCE`
  * `GENERATION_FAITHFULNESS`
  * `GENERATION_CORRECTNESS`
* Additional / System-level:

  * `LATENCY`
  * `DIVERSITY`
  * `NOISE_ROBUSTNESS`
  * `NEGATIVE_REJECTION`
  * `COUNTERFACTUAL_ROBUSTNESS`

### 2.2 Metric Modules

Add the following modules under `auepora_eval/metrics/`:

```text
metrics/
  __init__.py
  retrieval_basic.py        # Recall@k, Precision@k, HitRate@k
  retrieval_ranking.py      # MRR, MAP, nDCG, rank-based metrics
  generation_overlap.py     # Exact match, F1, ROUGE, BLEU
  generation_semantic.py    # BERTScore / embedding similarity
  generation_llm_judge.py   # LLM-based relevance / faithfulness / quality
  robustness.py             # Noise robustness, counterfactual robustness, negative rejection
  additional.py             # Latency, diversity
```

Each metric will be a class implementing `core.metrics_base.Metric`.

## 3. Retrieval Metrics

All retrieval metrics operate on:

* `EvaluationSample.relevant_docs: List[Document] | None`
* `EvaluationSample.candidate_docs: List[Document] | None` (for some metrics)
* `SystemOutputs.retrieved: List[RetrievedDocument]`

Each retrieved element includes `doc.doc_id`, `score`, and `rank`.

### 3.1 Retrieval Relevance Metrics (`RETRIEVAL_RELEVANCE`)

These treat retrieval as set/ordered set prediction of **relevant document IDs**.

#### 3.1.1 Recall@k

**Class**: `RecallAtK` (already sketched, keep as-is).

* **Module**: `metrics/retrieval_basic.py`
* **Target**: `TargetCategory.RETRIEVAL_RELEVANCE`
* **Config**:

  * `k: int`
* **Required fields**: `relevant_docs`
* **Definition per sample**:

  * Let `G` be the set of gold doc IDs from `relevant_docs`.
  * Let `R_k` be the set of doc IDs from top-k `SystemOutputs.retrieved`.
  * If `|G| = 0`, skip sample.
  * `recall_k = |G ∩ R_k| / |G|`.
* **Aggregation**: arithmetic mean over all valid samples.

Edge cases:

* If no valid samples (no sample with non-empty `relevant_docs`), metric value = `0.0`, `details["num_samples"] = 0`.

#### 3.1.2 Precision@k

**Class**: `PrecisionAtK`.

* **Module**: `metrics/retrieval_basic.py`
* **Target**: `RETRIEVAL_RELEVANCE`
* **Config**:

  * `k: int`
* **Required fields**: `relevant_docs`
* **Definition per sample**:

  * `G` as above, `R_k` as set of doc IDs from top-k retrieved.
  * `precision_k = |G ∩ R_k| / min(k, |R|)` where `R` is retrieved list (handle cases with fewer than k docs).
* **Aggregation**: mean of `precision_k` across samples (including samples with `|G| = 0` → precision defined as 0 if `G` empty? Implementation decision:

  * For v1, **skip** samples where `G` is empty (same as Recall@k), and record `details["skipped_empty_gold"]` = count.

#### 3.1.3 HitRate@k

**Class**: `HitRateAtK`.

* **Module**: `metrics/retrieval_basic.py`
* **Target**: `RETRIEVAL_RELEVANCE`
* **Config**:

  * `k: int`
* **Definition per sample**:

  * `hit_k = 1` if `G ∩ R_k` non-empty else `0` (only tests if at least one relevant doc appears in top-k).
* **Aggregation**: mean of `hit_k`.

### 3.2 Retrieval Ranking / Accuracy Metrics (`RETRIEVAL_ACCURACY`)

These assume we care about the **ranking quality** of results vs. relevant docs.

#### 3.2.1 Mean Reciprocal Rank (MRR@k)

**Class**: `MRRAtK`.

* **Module**: `metrics/retrieval_ranking.py`
* **Target**: `RETRIEVAL_ACCURACY`
* **Config**:

  * `k: int | None` – if `None`, use full ranking; otherwise truncate at k.
* **Required fields**: `relevant_docs`
* **Definition per sample**:

  * Let `rank_i` be the rank (1-based) of the first relevant doc in `SystemOutputs.retrieved`. If no relevant doc within top-k (or full list if `k=None`), contribution is 0.
  * `rr = 1 / rank_i` if found, else `0`.
* **Aggregation**: mean of `rr` across samples with `|G| > 0`.

#### 3.2.2 Mean Average Precision (MAP)

**Class**: `MeanAveragePrecision`.

* **Module**: `metrics/retrieval_ranking.py`
* **Target**: `RETRIEVAL_ACCURACY`
* **Config**:

  * `k: int | None` – cut-off for contributions, optional.
* **Required fields**: `relevant_docs`
* **Definition per sample**:

  * For each retrieved document at rank `r` (1-based), compute precision at `r` if the doc is relevant.
  * Average precision (AP) is mean of precision values at ranks where a new relevant doc is found.
  * If no relevant docs retrieved, AP = 0.
* **Aggregation**: mean AP over all samples with `|G| > 0`.

Implementation details:

* If `k` set, ignore documents after rank `k`.

#### 3.2.3 nDCG@k (normalized Discounted Cumulative Gain)

**Class**: `NDCGAtK`.

* **Module**: `metrics/retrieval_ranking.py`
* **Target**: `RETRIEVAL_ACCURACY`
* **Config**:

  * `k: int`
  * `gain_scheme: str = "binary"` (v1 only binary relevance; extension later for graded relevance using `labels` or doc metadata).
* **Required fields**: `relevant_docs`
* **Definition per sample (binary relevance)**:

  * For each rank `r` (1-based) up to `k`, relevance `rel_r` = 1 if doc at rank `r` is in gold `G`; 0 otherwise.
  * `DCG_k = Σ_{r = 1..k} (2^{rel_r} - 1) / log2(r + 1)`.
  * `IDCG_k` = DCG of ideal ranking where all relevant docs (up to k) are at the top.
  * `nDCG_k = DCG_k / IDCG_k` if `IDCG_k > 0` else `0`.
* **Aggregation**: mean of `nDCG_k` over samples with `|G| > 0`.

## 4. Generation Metrics

Generation metrics operate on:

* `EvaluationSample.query`
* `EvaluationSample.reference_answer`
* `EvaluationSample.relevant_docs` (for faithfulness)
* `SystemOutputs.response`

### 4.1 Exact Match and F1 (QA-style) – `GENERATION_CORRECTNESS`

#### 4.1.1 Exact Match (EM)

**Class**: `ExactMatch`.

* **Module**: `metrics/generation_overlap.py`
* **Target**: `GENERATION_CORRECTNESS`
* **Required fields**: `reference_answer`
* **Normalization**:

  * Lowercase.
  * Strip whitespace.
  * Remove basic punctuation (`.,!?;:"'` etc.).
  * Optionally normalize articles (`a`, `an`, `the`) – configurable.
* **Definition per sample**:

  * Let `y_pred` = normalized predicted text, `y_ref` = normalized reference text.
  * `em = 1` if `y_pred == y_ref`, else `0`.
* **Aggregation**: mean of `em` over samples with non-empty `reference_answer`.

Config:

* `ignore_case: bool = True`
* `ignore_punctuation: bool = True`
* `ignore_articles: bool = True`

#### 4.1.2 Token-level F1

**Class**: `TokenF1`.

* **Module**: `metrics/generation_overlap.py`
* **Target**: `GENERATION_CORRECTNESS`
* **Required fields**: `reference_answer`
* **Tokenization**: simple whitespace split after normalization (same as EM).
* **Definition per sample**:

  * Let `pred_tokens`, `ref_tokens` be token lists.
  * `common = intersection multiset size`.
  * `precision = common / len(pred_tokens)` (0 if empty).
  * `recall = common / len(ref_tokens)` (0 if empty).
  * `f1 = 2 * precision * recall / (precision + recall)` if both > 0 else 0.
* **Aggregation**: mean F1 across samples.

### 4.2 Overlap-based Similarity – `GENERATION_CORRECTNESS` / `GENERATION_RELEVANCE`

#### 4.2.1 ROUGE (1/2/L)

**Class**: `RougeScores`.

* **Module**: `metrics/generation_overlap.py` (or keep `RougeLAnswer` there and extend).
* **Target**:

  * Default: `GENERATION_CORRECTNESS` (vs reference answer).
  * Optionally allow `GENERATION_RELEVANCE` when comparing to query.
* **Dependencies**: `rouge-score` library.
* **Config**:

  * `variants: List[str] = ["rouge1", "rouge2", "rougeL"]`
  * `use_stemmer: bool = True`
  * `compare_to: str = "reference"` (either `"reference"` or `"query"`).
* **Definition per sample**:

  * Compare `SystemOutputs.response.text` to:

    * `reference_answer.text` if `compare_to="reference"`.
    * `EvaluationSample.query` if `compare_to="query"`.
  * Use F1 variant of ROUGE scores.
* **Output**:

  * For simplicity, we may define separate Metric classes per variant (e.g., `RougeLAnswer`), or `RougeScores` returning aggregated value for one variant.

Implementation option:

* Implement separate classes `Rouge1`, `Rouge2`, `RougeL`.

  * `name = "rouge1_answer"`, `"rouge2_answer"`, `"rougeL_answer"`.

#### 4.2.2 BLEU

**Class**: `BleuScore`.

* **Module**: `metrics/generation_overlap.py`
* **Target**: `GENERATION_CORRECTNESS`
* **Dependencies**: `sacrebleu` or NLTK BLEU. Prefer `sacrebleu` for reproducibility.
* **Config**:

  * `max_order: int = 4`
  * `smooth_method` as supported by library (e.g. `exp`, `floor`).
* **Definition per sample**:

  * BLEU between hypothesis (prediction) and a single reference.
  * Implementation may compute document-level BLEU across all samples (as sacrebleu does).

Simplification for v1:

* Compute corpus-level BLEU across all predictions and references (sacrebleu standard). MetricResult.value is that corpus-level score.
* `details` can include per-sample BLEU if needed later.

### 4.3 Semantic Similarity – `GENERATION_CORRECTNESS` / `GENERATION_RELEVANCE`

#### 4.3.1 BERTScore

**Class**: `BertScoreMetric`.

* **Module**: `metrics/generation_semantic.py`
* **Target**: `GENERATION_CORRECTNESS` (vs reference) or `GENERATION_RELEVANCE` (vs query), controlled by config.
* **Dependencies**: `bert-score` package.
* **Config**:

  * `model_type: str = "microsoft/deberta-xlarge-mnli"` (or similar default).
  * `num_layers: Optional[int]` (passed through to library).
  * `compare_to: str = "reference" | "query"`.
  * `score_type: str = "F1"` (can choose Precision/Recall/F1).
* **Definition**:

  * Use BERTScore library to compute similarity between each prediction and its reference string.
  * Aggregate by mean across samples.

#### 4.3.2 Embedding Cosine Similarity

**Class**: `EmbeddingSimilarity`.

* **Module**: `metrics/generation_semantic.py`
* **Target**: `GENERATION_CORRECTNESS` or `GENERATION_RELEVANCE`.
* **Inputs**:

  * External `embed_fn: Callable[[List[str]], List[List[float]]]` passed at metric init time.
* **Config**:

  * `compare_to: str = "reference" | "query"`.
* **Definition per sample**:

  * Compute embedding for prediction and comparison target.
  * Cosine similarity between vectors.
* **Aggregation**: mean cosine similarity.

### 4.4 Faithfulness Metrics – `GENERATION_FAITHFULNESS`

Faithfulness measures whether the answer is **supported by the retrieved evidence**.

We define two families: overlap-based and LLM-judged.

#### 4.4.1 Overlap-Based Faithfulness (Lexical Proxy)

**Class**: `EvidenceOverlapFaithfulness`.

* **Module**: `metrics/generation_overlap.py`
* **Target**: `GENERATION_FAITHFULNESS`
* **Required fields**:

  * `EvaluationSample.relevant_docs` OR `SystemOutputs.retrieved` (configurable which evidence source to use).
* **Config**:

  * `evidence_source: str = "retrieved" | "relevant"`.
  * `n: int = 1` (n-gram size, typically 1 or 2).
* **Definition per sample**:

  * Let `E` = concatenated text of either `relevant_docs` or top-k `retrieved` docs.
  * Tokenize prediction and evidence.
  * Compute `n`-gram recall of prediction w.r.t. evidence: fraction of prediction n-grams that appear in evidence.
* **Aggregation**: mean recall.

This is a crude proxy; mainly to implement something lightweight. For more accurate faithfulness, we rely on LLM-based metric below.

#### 4.4.2 LLM-Judged Faithfulness

**Class**: `LLMFaithfulnessScore`.

* **Module**: `metrics/generation_llm_judge.py`
* **Target**: `GENERATION_FAITHFULNESS`
* **Inputs**:

  * An `LLMCritic` (see §6.1) instance.
* **Required fields**:

  * `SystemOutputs.response`.
  * Evidence: either `relevant_docs` or `retrieved` docs.
* **Config**:

  * `evidence_source: "retrieved" | "relevant"`.
  * `scale: str = "0-1"` (expected score range from LLM; we will normalize to `[0,1]`).
* **Definition per sample**:

  * Build a prompt giving the LLM:

    * The question.
    * The answer.
    * The evidence snippets.
  * Ask to output a numeric faithfulness score in `[0, 1]` (or `[1, 5]` then normalized) and optionally a rationale.
  * Parse numeric score.
* **Aggregation**: mean of scores.

Implementation details:

* To keep metrics deterministic, all randomness should be controlled via the underlying LLM client (seed / temperature), or enforced in template (use temperature=0 where possible).

## 5. LLM-Judged Generation Quality / Relevance – `GENERATION_RELEVANCE`

#### 5.1 Answer Relevance / Quality

**Class**: `LLMAnswerQuality`.

* **Module**: `metrics/generation_llm_judge.py`
* **Target**: `GENERATION_RELEVANCE`
* **Inputs**:

  * `LLMCritic` instance.
* **Required fields**:

  * `EvaluationSample.query`
  * `SystemOutputs.response`
* **Config**:

  * `dimensions: List[str] = ["relevance", "helpfulness", "clarity"]`.
  * `aggregation: str = "mean"` or allow weighting of dimensions.
* **Definition**:

  * Prompt LLM with query and answer; ask for a numeric score representing how well the answer addresses the query.
  * Optionally ask for per-dimension scores and aggregate.
* **Aggregation**:

  * Mean of overall score across samples.

Implementation detail:

* For v1, implement simple single-score variant (0–1 or 1–5 normalized to 0–1).

## 6. Robustness and System-Level Metrics

These metrics operate over **subsets of samples** identified via `EvaluationSample.labels`.

Common patterns:

* Baseline vs. perturbed pairs using `labels["variant_of"]`.
* Scenario tags like `labels["scenario"] in {"paraphrase", "typo", "unanswerable", "counterfactual"}`.

### 6.1 Noise Robustness – `NOISE_ROBUSTNESS`

Measures performance degradation when queries are noisy (paraphrased, with typos, etc.).

**Class**: `NoiseRobustness`.

* **Module**: `metrics/robustness.py`
* **Target**: `NOISE_ROBUSTNESS`
* **Inputs**:

  * A **base metric** instance (e.g., `TokenF1`, `RecallAtK`).
* **Required fields**:

  * `labels["variant_of"]` linking noisy samples to their clean parent.
  * `labels["scenario"]` with values like `"paraphrase"`, `"typo"`.
* **Definition**:

  * Step 1: For all samples with **no** `variant_of` key → treat as base set.
  * Step 2: For all samples with `variant_of` and `scenario in noise_scenarios` → treat as noisy set.
  * Step 3: Compute base metric separately on base and noisy sets:

    * `base_score`, `noisy_score`.
  * Noise robustness score can be:

    * `robustness = noisy_score / base_score` if `base_score > 0` else 0.
* **Aggregation**: A single `MetricResult` with value = `robustness`, and `details` including `base_score`, `noisy_score`, sample counts.

### 6.2 Negative Rejection – `NEGATIVE_REJECTION`

Measures how well the system avoids hallucinating answers on **unanswerable** queries.

**Class**: `NegativeRejectionRate`.

* **Module**: `metrics/robustness.py`
* **Target**: `NEGATIVE_REJECTION`
* **Required fields**:

  * `labels["scenario"] == "unanswerable"` for negative samples.
  * `SystemOutputs.response.text`.
* **Assumption**:

  * Reference answer contains the expected abstention pattern, e.g., `"I don't know"` or a generic refusal. For v1, we keep this simple.
* **Config**:

  * `refusal_patterns: List[str] = ["i don't know", "cannot answer", "not enough information"]`.
* **Definition per **unanswerable** sample**:

  * Normalize prediction (lowercase, strip).
  * `is_rejected = any(pattern in prediction for pattern in refusal_patterns)`.
* **Metric value**:

  * `negative_rejection_rate = (# of unanswerable samples with is_rejected = True) / (# of unanswerable samples)`.

### 6.3 Counterfactual Robustness – `COUNTERFACTUAL_ROBUSTNESS`

Measures consistency between answers under small counterfactual variants of the query.

**Class**: `CounterfactualConsistency`.

* **Module**: `metrics/robustness.py`
* **Target**: `COUNTERFACTUAL_ROBUSTNESS`
* **Inputs**:

  * A **base semantic similarity metric** (e.g., `EmbeddingSimilarity` or `BertScoreMetric`).
* **Required fields**:

  * For each counterfactual sample, `labels["variant_of"]` linking to base sample.
  * `labels["scenario"] == "counterfactual"` (or similar).
* **Definition**:

  * For each pair `(base_sample, cf_sample)`:

    * Compute similarity between their answers using provided base metric.
  * Robustness score = mean similarity across all such pairs.

Implementation detail:

* Provide an internal helper to map `variant_of` IDs to base sample indices.

## 7. Latency and Diversity Metrics (Additional)

### 7.1 Latency – `LATENCY`

We already have `MeanLatency`; we extend with distribution-oriented metrics.

#### 7.1.1 MeanLatency (existing)

Keep as specified.

#### 7.1.2 QuantileLatency

**Class**: `QuantileLatency`.

* **Module**: `metrics/additional.py`
* **Target**: `LATENCY`
* **Config**:

  * `timing_key: str = "end_to_end"`
  * `quantile: float = 0.95` (e.g., 0.9, 0.95, 0.99).
* **Definition**:

  * Collect all `out.timings[timing_key]`.
  * Sort ascending.
  * Return value at position `ceil(q * n) - 1` (0-based index), where `q` is quantile, `n` is number of values.
* **Details**:

  * `details["num_samples"]`, `details["quantile"]`.

### 7.2 Diversity – `DIVERSITY`

We consider both **generation diversity** (variety of answers) and **retrieval diversity** (variety of docs).

#### 7.2.1 Text Diversity via Distinct-n

**Class**: `DistinctN`.

* **Module**: `metrics/additional.py`
* **Target**: `DIVERSITY`
* **Config**:

  * `n: int = 2`
* **Definition**:

  * Across all predictions, gather all `n`-grams.
  * `distinct_n = (# unique n-grams) / (# total n-grams)`.

#### 7.2.2 Retrieval Diversity – Intra-list Diversity

**Class**: `IntraListDiversity`.

* **Module**: `metrics/additional.py`
* **Target**: `DIVERSITY`
* **Inputs**:

  * Document embedding function or pre-computed embeddings accessible via `Document.metadata["embedding"]`.
* **Config**:

  * `top_k: int = 5`
* **Definition per sample**:

  * For top-k retrieved docs, compute pairwise cosine similarities.
  * `ILD = 1 - average_cosine_similarity`.
* **Aggregation**:

  * Mean ILD across samples.

Implementation detail:

* For v1, assume embeddings are present in `doc.metadata["embedding"]` as a list of floats; if missing, skip sample.

## 8. LLM Critic Interface (`generation_llm_judge.py`)

Define a shared critic interface for all LLM-based metrics.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMCritic(ABC):
    @abstractmethod
    def score(self, *, prompt: str, metadata: Dict[str, Any] | None = None) -> float:
        """Return a numeric score (typically in [0, 1] or [1, 5]) for the given prompt.

        The implementation handles calling an LLM, parsing output, and mapping to a float.
        """
        raise NotImplementedError
```

All LLM-based metrics (`LLMFaithfulnessScore`, `LLMAnswerQuality`, etc.) accept an `LLMCritic` instance at initialization.

Implementation suggestion:

* Provide a simple reference implementation in `examples/` that uses a particular LLM provider; keep core library abstract.

## 9. Metric Base Class Implementation Notes

Recall `core.metrics_base.Metric`:

```python
class Metric(ABC):
    name: str
    target: TargetCategory

    @abstractmethod
    def required_fields(self) -> List[str]:
        ...

    @abstractmethod
    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        ...
```

### 9.1 General Implementation Rules

1. **Length checks**: all `compute` implementations must assert `len(samples) == len(outputs)`.
2. **Skipping samples**:

   * If a sample doesn’t have required ground truth (e.g., `reference_answer is None`), the metric may **skip** it.
   * Skipped samples should not contribute to the numeric value.
   * Record counts in `MetricResult.details["num_samples"]` (actually evaluated) and optionally `"num_skipped"`.
3. **Empty support**:

   * If no samples are evaluated (e.g., no `reference_answer` for any sample), return value = `0.0` and `details["num_samples"] = 0`.
4. **Determinism**:

   * Metrics should be deterministic given the same inputs.
   * Any stochastic components (e.g., LLM) should be controlled from outside via deterministic configurations.
5. **Naming**:

   * `name` must be stable and machine-readable (e.g., `"recall@5"`, `"token_f1"`, `"llm_faithfulness"`).

### 9.2 Dataset Slicing Helper (Optional v1.1)

Later we can introduce a generic `SlicedMetric` wrapper that:

* Takes a base metric and a predicate on `EvaluationSample`.
* Computes the metric only on that subset.

This is not required for v1 but is a natural extension for more complex analyses.

## 10. Module-by-Module TODOs

### 10.1 `metrics/retrieval_basic.py`

Implement:

* [ ] `RecallAtK`
* [ ] `PrecisionAtK`
* [ ] `HitRateAtK`

### 10.2 `metrics/retrieval_ranking.py`

Implement:

* [ ] `MRRAtK`
* [ ] `MeanAveragePrecision`
* [ ] `NDCGAtK`

### 10.3 `metrics/generation_overlap.py`

Implement:

* [ ] `ExactMatch`
* [ ] `TokenF1`
* [ ] `Rouge1` / `Rouge2` / `RougeL`
* [ ] `EvidenceOverlapFaithfulness`

### 10.4 `metrics/generation_semantic.py`

Implement:

* [ ] `BertScoreMetric`
* [ ] `EmbeddingSimilarity`

### 10.5 `metrics/generation_llm_judge.py`

Implement:

* [ ] `LLMCritic` abstract base class
* [ ] `LLMFaithfulnessScore`
* [ ] `LLMAnswerQuality`

### 10.6 `metrics/robustness.py`

Implement:

* [ ] `NoiseRobustness`
* [ ] `NegativeRejectionRate`
* [ ] `CounterfactualConsistency`

### 10.7 `metrics/additional.py`

Implement:

* [ ] `MeanLatency` (existing spec)
* [ ] `QuantileLatency`
* [ ] `DistinctN`
* [ ] `IntraListDiversity`

## 11. Testing Guidelines

For each metric, add unit tests in `tests/metrics/` that:

1. **Check basic correctness** on small hand-constructed examples.
2. **Check edge cases**:

   * No relevant docs.
   * No reference answer.
   * Empty predictions.
3. **Check aggregation**:

   * Known input → known aggregate (e.g., recall/precision on small rankings).
4. **For external-dependency metrics** (ROUGE, BLEU, BERTScore):

   * Use fixed small examples and snapshot the numeric scores with a tolerance (e.g., `abs(a - b) < 1e-6`).

This spec gives the engineer a complete roadmap for implementing the metric layer of the Auepora evaluation library, fully covering retrieval, generation, robustness, and system-level targets.
