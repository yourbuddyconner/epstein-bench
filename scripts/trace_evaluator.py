"""
TRACe Benchmark Evaluator.
Evaluates RAG systems using the TRACe framework (Reasoning, Adherence, Completeness).
Uses LLM-as-a-judge for granular metrics.
"""

import json
import time
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()


@dataclass
class TRACeMetrics:
    """TRACe evaluation metrics."""
    relevance: float = 0.0      # (Retriever) Is the retrieved context relevant?
    utilization: float = 0.0    # (Generator) Did the model use the context?
    adherence: float = 0.0      # (Generator) Is the answer supported by context? (Hallucination check)
    completeness: float = 0.0   # (Generator) Does it fully answer the question?
    correctness: float = 0.0    # (End-to-End) Does it match the ground truth meaning?
    reasoning: str = ""         # LLM explanation


@dataclass
class EvaluationResult:
    """Results for a single question evaluation."""
    question_id: str
    question: str
    correct_answer: str
    predicted_answer: str
    retrieved_docs: List[str]
    retrieved_context_text: str
    trace_metrics: TRACeMetrics
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


class TraceEvaluator:
    """Evaluates RAG systems on the TRACe benchmark."""
    
    def __init__(self, benchmark_file: str, results_dir: str = "./results", 
                 model_name: str = "gpt-4o"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        
        # Load benchmark
        with open(benchmark_file, 'r') as f:
            self.benchmark = json.load(f)
        
        self.questions = self.benchmark['questions']
        self.metadata = self.benchmark['metadata']
        
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. TRACe evaluation will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            
        # Fallback embedding model
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Loaded benchmark: {self.metadata.get('name', 'Unknown')}")
        print(f"Questions: {len(self.questions)}")
    
    def evaluate_rag_system(self, rag_system: Any, 
                           system_name: str = "baseline",
                           limit: int = None) -> Dict:
        """Evaluate a RAG system."""
        print(f"\nEvaluating {system_name} on TRACe benchmark...")
        
        results = []
        questions_to_eval = self.questions[:limit] if limit else self.questions
        
        for question in tqdm(questions_to_eval, desc="Evaluating"):
            start_time = time.time()
            
            try:
                # Call the RAG system
                # Expecting: answer, ids, timing OR answer, ids, extra_dict
                # Ideally, rag_system returns the context it used in the 3rd arg
                answer, retrieved_docs, extra = rag_system.retrieve_and_answer(
                    question['question']
                )
                
                total_time = (time.time() - start_time) * 1000
                
                # Extract context text if provided, else empty
                context_text = ""
                if isinstance(extra, dict):
                    context_text = extra.get('context_text', '')
                
                # Calculate TRACe metrics
                trace = self._calculate_trace_metrics(
                    question['question'],
                    question['answer'],
                    answer,
                    context_text
                )
                
                result = EvaluationResult(
                    question_id=question['question_id'],
                    question=question['question'],
                    correct_answer=question['answer'],
                    predicted_answer=answer,
                    retrieved_docs=retrieved_docs,
                    retrieved_context_text=context_text,
                    trace_metrics=trace,
                    retrieval_time_ms=extra.get('retrieval_ms', 0) if isinstance(extra, dict) else 0,
                    generation_time_ms=extra.get('generation_ms', 0) if isinstance(extra, dict) else 0,
                    total_time_ms=total_time
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {question['question_id']}: {e}")
        
        metrics = self._aggregate_metrics(results, system_name)
        self._save_results(system_name, metrics, results)
        self._print_summary(metrics)
        
        return metrics
    
    def _calculate_trace_metrics(self, question: str, gold_answer: str, 
                               pred_answer: str, context: str) -> TRACeMetrics:
        """Calculate TRACe metrics using LLM Judge."""
        
        if not self.client:
            return self._fallback_metrics(gold_answer, pred_answer)
        
        try:
            prompt = f"""You are an expert AI Judge for the TRACe RAG benchmark.
Evaluate the System Response.

Input:
- Question: {question}
- Retrieved Context (what the system saw): {context[:2500]}...
- System Response: {pred_answer}
- Gold Answer: {gold_answer}

Scores (0.0 to 1.0):
1. Relevance: Is the Context relevant to the Question?
2. Adherence: Is the Response supported by the Context? (1.0 = fully supported, 0.0 = hallucination). 
   Note: If context is missing/irrelevant and model says "I don't know", Adherence = 1.0.
3. Completeness: Does the Response fully answer the Question compared to the Gold Answer?
4. Correctness: Is the Response factually correct based on the Gold Answer?

Output JSON:
{{
  "relevance": float,
  "adherence": float,
  "completeness": float,
  "correctness": float,
  "reasoning": "concise explanation"
}}"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a strict and fair evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            
            return TRACeMetrics(
                relevance=data.get('relevance', 0.0),
                adherence=data.get('adherence', 0.0),
                completeness=data.get('completeness', 0.0),
                correctness=data.get('correctness', 0.0),
                reasoning=data.get('reasoning', "")
            )
            
        except Exception as e:
            print(f"LLM Judge error: {e}")
            return self._fallback_metrics(gold_answer, pred_answer)

    def _fallback_metrics(self, gold: str, pred: str) -> TRACeMetrics:
        """Cosine similarity fallback."""
        if not pred: return TRACeMetrics()
        
        g_emb = self.embed_model.encode([gold])
        p_emb = self.embed_model.encode([pred])
        sim = cosine_similarity(g_emb, p_emb)[0][0]
        
        return TRACeMetrics(
            correctness=float(sim),
            reasoning="Fallback: Cosine Similarity"
        )
    
    def _aggregate_metrics(self, results: List[EvaluationResult], system_name: str) -> Dict:
        if not results: return {}
        
        return {
            'system_name': system_name,
            'timestamp': datetime.now().isoformat(),
            'count': len(results),
            'correctness': np.mean([r.trace_metrics.correctness for r in results]),
            'relevance': np.mean([r.trace_metrics.relevance for r in results]),
            'adherence': np.mean([r.trace_metrics.adherence for r in results]),
            'completeness': np.mean([r.trace_metrics.completeness for r in results]),
            'latency': np.mean([r.total_time_ms for r in results]),
        }

    def _save_results(self, system_name: str, metrics: Dict, results: List[EvaluationResult]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(self.results_dir / f"{system_name}_metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
            
        with open(self.results_dir / f"{system_name}_trace_details_{timestamp}.jsonl", 'w') as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + '\n')

    def _print_summary(self, metrics: Dict) -> None:
        print("\n" + "="*60)
        print(f"TRACe SUMMARY - {metrics['system_name']}")
        print("="*60)
        print(f"Correctness:  {metrics['correctness']:.2%}")
        print(f"Adherence:    {metrics['adherence']:.2%}")
        print(f"Completeness: {metrics['completeness']:.2%}")
        print(f"Relevance:    {metrics['relevance']:.2%}")
        print(f"Latency:      {metrics['latency']:.1f}ms")


if __name__ == "__main__":
    evaluator = TraceEvaluator("./benchmarks/epstein_trace_v1.json")
    print("Evaluator ready.")
