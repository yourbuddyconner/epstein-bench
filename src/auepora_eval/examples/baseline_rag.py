import argparse
import logging
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.types import Document, EvaluationSample, Response, SystemOutputs, RetrievedDocument
from ..core.dataset import InMemoryDataset
from ..core.rag_interface import RAGSystem, Retriever, Generator
from ..core.runner import AueporaEvaluator
from ..io.jsonl import load_jsonl_dataset
from ..taskgen.openai_client import OpenAIClient
from ...config import config
from ...epstein_bench.plan import build_metric_plan
from ...epstein_bench.llm_critic import OpenAIChatCritic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TfidfRetriever(Retriever):
    """A simple TF-IDF retriever."""
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Collect all text for indexing
        corpus_texts = [d.text for d in documents]
        if not corpus_texts:
            logger.warning("TfidfRetriever initialized with empty corpus.")
            self.matrix = None
        else:
            self.matrix = self.vectorizer.fit_transform(corpus_texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if self.matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        # Compute similarities
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Get top k indices
        top_indices = sims.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx]
            score = float(sims[idx])
            results.append(RetrievedDocument(doc=doc, score=score, rank=rank + 1))
            
        return results


class SimpleLLMGenerator(Generator):
    """A generator that uses an LLM to answer questions based on context."""
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client

    def generate(self, query: str, context_docs: List[Document]) -> Response:
        # Prepare context
        context_text = "\n\n".join([f"Doc {i+1}: {d.text}" for i, d in enumerate(context_docs)])
        
        if self.client:
            prompt = (
                "Answer the question based on the context provided below.\n"
                "If the answer is not in the context, say 'I cannot answer this based on the context.'\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )
            answer_text = self.client.generate(prompt).strip()
        else:
            # Fallback if no LLM client: just return the first doc's first sentence as a dummy answer
            if context_docs:
                answer_text = context_docs[0].text.split('.')[0] + "."
            else:
                answer_text = "No context available."
                
        return Response(text=answer_text)


class BaselineRAG(RAGSystem):
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator

    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        retrieved = self.retriever.retrieve(sample.query, top_k=top_k)
        docs = [r.doc for r in retrieved]
        response = self.generator.generate(sample.query, docs)
        return SystemOutputs(retrieved=retrieved, response=response)


def main():
    parser = argparse.ArgumentParser(description="Run Baseline RAG on a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--use_llm", action="store_true", help="Use OpenAI for generation (requires .env config)")
    parser.add_argument("--enable_bertscore", action="store_true", help="Include BERTScore metrics (requires bert-score package/models)")
    parser.add_argument("--enable_llm_metrics", action="store_true", help="Include LLM-judged metrics (requires OpenAI key)")
    args = parser.parse_args()

    # 1. Load Dataset
    logger.info(f"Loading dataset from {args.dataset}...")
    dataset = load_jsonl_dataset(args.dataset, name="baseline_eval")
    logger.info(f"Loaded {len(dataset)} samples.")

    # 2. Build Corpus for Retriever
    # In a real scenario, we'd load the full corpus. 
    # For this baseline script, we'll construct the corpus from the 'relevant_docs' 
    # available in the dataset itself to ensure we have something to search over.
    # This is a "closed-world" evaluation relative to the dataset's gold docs 
    # (plus distractors if we had them). 
    # To do a proper open-retrieval test, we'd need the full corpus loaded separately.
    # For simplicity here, we aggregate all docs found in the evaluation dataset.
    
    all_docs_map = {}
    for sample in dataset:
        if sample.relevant_docs:
            for d in sample.relevant_docs:
                all_docs_map[d.doc_id] = d
        # Add candidate docs if present
        if sample.candidate_docs:
            for d in sample.candidate_docs:
                all_docs_map[d.doc_id] = d
                
    corpus_docs = list(all_docs_map.values())
    logger.info(f"Constructed retrieval corpus with {len(corpus_docs)} unique documents from dataset.")

    if not corpus_docs:
        logger.error("No documents found in dataset to build index. Ensure dataset has 'relevant_docs'.")
        return

    # 3. Initialize Components
    logger.info("Initializing TF-IDF Retriever...")
    retriever = TfidfRetriever(corpus_docs)
    
    llm_client = None
    if args.use_llm:
        if config.openai_api_key:
            logger.info(f"Initializing OpenAI Generator ({config.tagging_model_name})...")
            llm_client = OpenAIClient(
                api_key=config.openai_api_key,
                model=config.tagging_model_name,
                max_tokens=config.tagging_max_output_tokens
            )
        else:
            logger.warning("OpenAI API key not found. Falling back to dummy generator.")
    
    generator = SimpleLLMGenerator(client=llm_client)
    rag_system = BaselineRAG(retriever, generator)

    llm_metric_critic = None
    if args.enable_llm_metrics:
        if not config.openai_api_key:
            logger.error("LLM metrics requested but OPENAI_API_KEY is not set.")
            return
        llm_metric_critic = OpenAIChatCritic()

    plan = build_metric_plan(
        top_k=args.top_k,
        enable_bertscore=args.enable_bertscore,
        enable_llm_metrics=args.enable_llm_metrics,
        llm_critic=llm_metric_critic,
    )

    # 5. Run Evaluation
    logger.info("Starting Evaluation...")
    evaluator = AueporaEvaluator(system=rag_system, plan=plan, top_k=args.top_k)
    results = evaluator.evaluate(dataset)

    # 6. Report Results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    for r in results:
        print(f"{r.name:<20} | {r.value:.4f} | Details: {r.details}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()

