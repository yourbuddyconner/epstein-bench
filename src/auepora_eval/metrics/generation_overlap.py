from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from rouge_score import rouge_scorer

from ..core.metrics_base import Metric
from ..core.types import (
    EvaluationSample,
    SystemOutputs,
    MetricResult,
    TargetCategory,
    Document,
)


_PUNCT_RE = re.compile(r"[^\w\s]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize(text: str, ignore_case: bool, ignore_punct: bool, ignore_articles: bool) -> str:
    if ignore_case:
        text = text.lower()
    if ignore_punct:
        text = _PUNCT_RE.sub(" ", text)
    if ignore_articles:
        text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


class ExactMatch(Metric):
    def __init__(
        self,
        *,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        ignore_articles: bool = True,
    ):
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles
        self.name = "exact_match"
        self.target = TargetCategory.GENERATION_CORRECTNESS

    def required_fields(self) -> List[str]:
        return ["reference_answer"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []

        for sample, out in zip(samples, outputs):
            if not sample.reference_answer or not sample.reference_answer.text:
                continue
            pred = _normalize(
                out.response.text,
                self.ignore_case,
                self.ignore_punctuation,
                self.ignore_articles,
            )
            ref = _normalize(
                sample.reference_answer.text,
                self.ignore_case,
                self.ignore_punctuation,
                self.ignore_articles,
            )
            scores.append(1.0 if pred == ref else 0.0)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class TokenF1(Metric):
    def __init__(
        self,
        *,
        ignore_case: bool = True,
        ignore_punctuation: bool = True,
        ignore_articles: bool = True,
    ):
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.ignore_articles = ignore_articles
        self.name = "token_f1"
        self.target = TargetCategory.GENERATION_CORRECTNESS

    def required_fields(self) -> List[str]:
        return ["reference_answer"]

    def _tokens(self, text: str) -> List[str]:
        norm = _normalize(
            text,
            self.ignore_case,
            self.ignore_punctuation,
            self.ignore_articles,
        )
        return norm.split()

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []

        for sample, out in zip(samples, outputs):
            if not sample.reference_answer or not sample.reference_answer.text:
                continue

            pred_tokens = self._tokens(out.response.text)
            ref_tokens = self._tokens(sample.reference_answer.text)
            if not pred_tokens and not ref_tokens:
                scores.append(1.0)
                continue

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())
            if num_common == 0:
                scores.append(0.0)
                continue

            precision = num_common / len(pred_tokens) if pred_tokens else 0.0
            recall = num_common / len(ref_tokens) if ref_tokens else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class _RougeBase(Metric):
    def __init__(self, variant: str, *, compare_to: str = "reference", use_stemmer: bool = True):
        assert compare_to in {"reference", "query"}
        self.variant = variant
        self.compare_to = compare_to
        self.use_stemmer = use_stemmer
        target_suffix = "answer" if compare_to == "reference" else "query"
        self.name = f"{variant}_{target_suffix}"
        self.target = TargetCategory.GENERATION_CORRECTNESS if compare_to == "reference" else TargetCategory.GENERATION_RELEVANCE
        self._scorer = rouge_scorer.RougeScorer([variant], use_stemmer=use_stemmer)

    def required_fields(self) -> List[str]:
        return ["reference_answer"] if self.compare_to == "reference" else []

    def _reference_text(self, sample: EvaluationSample) -> Optional[str]:
        if self.compare_to == "reference":
            if not sample.reference_answer:
                return None
            return sample.reference_answer.text
        return sample.query

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []
        skipped = 0

        for sample, out in zip(samples, outputs):
            ref = self._reference_text(sample)
            if not ref:
                skipped += 1
                continue
            score = self._scorer.score(ref, out.response.text)[self.variant].fmeasure
            scores.append(score)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores), "num_skipped": skipped},
        )


class Rouge1(_RougeBase):
    def __init__(self, **kwargs):
        super().__init__("rouge1", **kwargs)


class Rouge2(_RougeBase):
    def __init__(self, **kwargs):
        super().__init__("rouge2", **kwargs)


class RougeL(_RougeBase):
    def __init__(self, **kwargs):
        super().__init__("rougeL", **kwargs)


class EvidenceOverlapFaithfulness(Metric):
    def __init__(self, *, evidence_source: str = "retrieved", n: int = 1, top_k: Optional[int] = None):
        assert evidence_source in {"retrieved", "relevant"}
        assert n >= 1
        self.evidence_source = evidence_source
        self.n = n
        self.top_k = top_k
        self.name = f"evidence_overlap_n{n}"
        self.target = TargetCategory.GENERATION_FAITHFULNESS

    def required_fields(self) -> List[str]:
        if self.evidence_source == "relevant":
            return ["relevant_docs"]
        return []

    def _evidence_docs(self, sample: EvaluationSample, output: SystemOutputs) -> List[Document]:
        if self.evidence_source == "relevant":
            return sample.relevant_docs or []
        docs = [r.doc for r in output.retrieved]
        if self.top_k is not None:
            docs = docs[: self.top_k]
        return docs

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Counter:
        if len(tokens) < n:
            return Counter()
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []
        for sample, out in zip(samples, outputs):
            evidence_docs = self._evidence_docs(sample, out)
            if not evidence_docs:
                continue

            evidence_text = " ".join(doc.text for doc in evidence_docs)
            pred_tokens = out.response.text.split()
            evidence_tokens = evidence_text.split()
            pred_ngrams = self._ngrams(pred_tokens, self.n)
            evidence_ngrams = self._ngrams(evidence_tokens, self.n)

            if not pred_ngrams:
                scores.append(0.0)
                continue

            overlap = sum((pred_ngrams & evidence_ngrams).values())
            total = sum(pred_ngrams.values())
            scores.append(overlap / total if total > 0 else 0.0)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class BleuScore(Metric):
    def __init__(self, *, max_order: int = 4, smooth_method: str = "exp"):
        self.max_order = max_order
        self.smooth_method = smooth_method
        self.name = f"bleu_{max_order}"
        self.target = TargetCategory.GENERATION_CORRECTNESS

    def required_fields(self) -> List[str]:
        return ["reference_answer"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        refs: List[str] = []
        hyps: List[str] = []

        for sample, out in zip(samples, outputs):
            if not sample.reference_answer:
                continue
            refs.append(sample.reference_answer.text)
            hyps.append(out.response.text)

        if not refs:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0},
            )

        try:
            from sacrebleu.metrics import BLEU
        except ImportError:
            if not getattr(self, "_warned_missing_pkg", False):
                self._warned_missing_pkg = True
                from warnings import warn

                warn("sacrebleu not installed; BLEU metric will output 0. Install via `pip install sacrebleu`.")
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": len(refs), "skipped": True},
            )

        bleu = BLEU(
            smooth_method=self.smooth_method,
            max_ngram_order=self.max_order,
            effective_order=True,
        )
        score = bleu.corpus_score(hyps, [refs])

        return MetricResult(
            name=self.name,
            target=self.target,
            value=float(score.score / 100.0),
            details={"num_samples": len(refs)},
        )

