from concurrent.futures import ThreadPoolExecutor
from typing import List

from tqdm import tqdm

from .dataset import EvaluationDataset
from .rag_interface import RAGSystem
from .plan import EvaluationPlan
from .types import EvaluationSample, SystemOutputs, MetricResult


class AueporaEvaluator:
    def __init__(self, system: RAGSystem, plan: EvaluationPlan, *, top_k: int = 5, max_workers: int = 5):
        self.system = system
        self.plan = plan
        self.top_k = top_k
        self.max_workers = max_workers

    def evaluate(self, dataset: EvaluationDataset) -> List[MetricResult]:
        samples: List[EvaluationSample] = []
        
        # Collect all samples first
        for sample in dataset:
            samples.append(sample)
            
        outputs: List[SystemOutputs] = [None] * len(samples) # type: ignore

        def process_sample(idx_and_sample):
            idx, s = idx_and_sample
            return idx, self.system.run(s, top_k=self.top_k)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(lambda s: self.system.run(s, top_k=self.top_k), samples)
            # Wrap iterator with tqdm for progress bar
            outputs = list(tqdm(results, total=len(samples), desc="Evaluating RAG System"))

        # Auepora Dataset step – validate annotations vs. planned Metrics
        self.plan.validate_dataset(samples)

        # Auepora Metric step – compute results
        results: List[MetricResult] = []
        for metric in self.plan.metrics:
            res = metric.compute(samples, outputs)
            results.append(res)

        return results

