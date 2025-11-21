from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
from openai import OpenAI

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory
from ...config import config


class BertScoreMetric(Metric):
    def __init__(
        self,
        *,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        num_layers: Optional[int] = None,
        compare_to: str = "reference",
        score_type: str = "f1",
        use_openai: bool = False,
    ):
        assert compare_to in {"reference", "query"}
        assert score_type.lower() in {"precision", "recall", "f1"}
        self.model_type = model_type
        self.num_layers = num_layers
        self.compare_to = compare_to
        self.score_type = score_type.lower()
        self.use_openai = use_openai
        self.name = f"{'openai_' if use_openai else ''}bertscore_{self.score_type}_{compare_to}"
        self.target = (
            TargetCategory.GENERATION_CORRECTNESS
            if compare_to == "reference"
            else TargetCategory.GENERATION_RELEVANCE
        )

    def required_fields(self) -> List[str]:
        return ["reference_answer"] if self.compare_to == "reference" else []

    def _refs(self, sample: EvaluationSample) -> Optional[str]:
        if self.compare_to == "reference":
            if not sample.reference_answer:
                return None
            return sample.reference_answer.text
        return sample.query

    def _score_batch_openai(self, text_pairs: List[tuple[str, str]]) -> List[float]:
        if not config.openai_api_key:
            raise RuntimeError("OpenAI API key not found in environment.")
            
        client = OpenAI(api_key=config.openai_api_key)
        
        # Prepare flattened list for embedding API
        flat_texts = []
        for a, b in text_pairs:
            flat_texts.append(a)
            flat_texts.append(b)
            
        # Call embedding API in batches if needed (OpenAI limit is ~2048 texts)
        # We can parallelize batch requests.
        batch_size = 1000
        all_embeddings = [None] * len(flat_texts) # Pre-allocate
        
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_batch(start_idx, batch_texts):
            try:
                resp = client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small"
                )
                return start_idx, [d.embedding for d in resp.data]
            except Exception as e:
                # Simple retry logic could go here
                raise e

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(0, len(flat_texts), batch_size):
                batch = flat_texts[i : i + batch_size]
                futures.append(executor.submit(fetch_batch, i, batch))
            
            for future in as_completed(futures):
                idx, embeddings = future.result()
                all_embeddings[idx : idx + len(embeddings)] = embeddings
            
        scores = []
        for i in range(0, len(all_embeddings), 2):
            vec_a = np.array(all_embeddings[i])
            vec_b = np.array(all_embeddings[i+1])
            
            denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
            score = 0.0 if denom == 0 else float(np.dot(vec_a, vec_b) / denom)
            scores.append(score)
            
        return scores

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        refs: List[str] = []
        hyps: List[str] = []
        for sample, out in zip(samples, outputs):
            ref = self._refs(sample)
            if not ref:
                continue
            refs.append(ref)
            hyps.append(out.response.text)

        if not refs:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0},
            )

        if self.use_openai:
            pairs = list(zip(hyps, refs))
            values = np.array(self._score_batch_openai(pairs))
        else:
            values = np.array([self._score_pair(h, r) for h, r in zip(hyps, refs)])

        value = float(values.mean())
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(values)},
        )


class EmbeddingSimilarity(Metric):
    def __init__(
        self,
        embed_fn: Callable[[List[str]], List[List[float]]],
        *,
        compare_to: str = "reference",
    ):
        assert compare_to in {"reference", "query"}
        self.embed_fn = embed_fn
        self.compare_to = compare_to
        self.name = f"embedding_similarity_{compare_to}"
        self.target = (
            TargetCategory.GENERATION_CORRECTNESS
            if compare_to == "reference"
            else TargetCategory.GENERATION_RELEVANCE
        )

    def required_fields(self) -> List[str]:
        return ["reference_answer"] if self.compare_to == "reference" else []

    def _refs(self, sample: EvaluationSample) -> Optional[str]:
        if self.compare_to == "reference":
            if not sample.reference_answer:
                return None
            return sample.reference_answer.text
        return sample.query

    @staticmethod
    def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def pair_score(self, text_a: str, text_b: str) -> float:
        emb = np.array(self.embed_fn([text_a, text_b]))
        return self._cosine(emb[0], emb[1])

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        hyps: List[str] = []
        refs: List[str] = []

        for sample, out in zip(samples, outputs):
            ref = self._refs(sample)
            if not ref:
                continue
            refs.append(ref)
            hyps.append(out.response.text)

        if not refs:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0},
            )

        scores = [self.pair_score(h, r) for h, r in zip(hyps, refs)]

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )
