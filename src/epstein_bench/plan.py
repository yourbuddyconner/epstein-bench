from __future__ import annotations

from typing import List, Optional

from sklearn.feature_extraction.text import HashingVectorizer

from ..auepora_eval.metrics import (
    RecallAtK,
    PrecisionAtK,
    HitRateAtK,
    MRRAtK,
    MeanAveragePrecision,
    NDCGAtK,
    ExactMatch,
    TokenF1,
    Rouge1,
    Rouge2,
    RougeL,
    BleuScore,
    EvidenceOverlapFaithfulness,
    EmbeddingSimilarity,
    BertScoreMetric,
    LLMFaithfulnessScore,
    LLMAnswerQuality,
)
from ..auepora_eval.core.plan import EvaluationPlan
from ..auepora_eval.metrics.generation_llm_judge import LLMCritic


def _hashing_embed_fn(n_features: int = 2**12):
    vectorizer = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
    )

    def embed(texts: List[str]):
        matrix = vectorizer.transform(texts)
        return matrix.toarray().tolist()

    return embed


def build_metric_plan(
    *,
    top_k: int,
    enable_bertscore: bool = False,
    use_openai_bertscore: bool = False,
    enable_llm_metrics: bool = False,
    llm_critic: Optional[LLMCritic] = None,
) -> EvaluationPlan:
    metrics = [
        RecallAtK(k=top_k),
        PrecisionAtK(k=top_k),
        HitRateAtK(k=top_k),
        MRRAtK(k=top_k),
        MeanAveragePrecision(k=top_k),
        NDCGAtK(k=top_k),
        ExactMatch(),
        TokenF1(),
        Rouge1(),
        Rouge2(),
        RougeL(),
        BleuScore(),
        EvidenceOverlapFaithfulness(evidence_source="retrieved", n=1, top_k=top_k),
    ]

    embed_fn = _hashing_embed_fn()
    metrics.extend(
        [
            EmbeddingSimilarity(embed_fn, compare_to="reference"),
            EmbeddingSimilarity(embed_fn, compare_to="query"),
        ]
    )

    if enable_bertscore:
        metrics.append(BertScoreMetric(compare_to="reference", use_openai=use_openai_bertscore))
        metrics.append(BertScoreMetric(compare_to="query", use_openai=use_openai_bertscore))

    if enable_llm_metrics:
        if not llm_critic:
            raise ValueError("LLM critic must be provided when enable_llm_metrics=True.")
        metrics.append(LLMFaithfulnessScore(llm_critic, evidence_source="retrieved", top_k=top_k))
        metrics.append(LLMAnswerQuality(llm_critic))

    return EvaluationPlan(metrics=metrics)

