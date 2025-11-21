import logging
import os
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.types import EvaluationSample, Document, Response, RetrievedDocument, SystemOutputs
from ..core.rag_interface import RAGSystem, Retriever, Generator
from ..taskgen.hf_adapter import HFCorpusAdapter
from ..taskgen.config import HFCorpusConfig, ChunkingConfig
from ..taskgen.preprocess import chunk_documents
from ..taskgen.openai_client import OpenAIClient
from ...config import config

logger = logging.getLogger(__name__)

class TfidfRetriever(Retriever):
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words='english')
        logger.info(f"Indexing {len(documents)} chunks with TF-IDF...")
        corpus_texts = [d.text for d in documents]
        if not corpus_texts:
            self.matrix = None
        else:
            self.matrix = self.vectorizer.fit_transform(corpus_texts)
        logger.info("Indexing complete.")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDocument]:
        if self.matrix is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        
        # Get top k indices
        # Note: argsort returns indices that sort the array. [::-1] reverses it.
        top_indices = sims.argsort()[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            doc = self.documents[idx]
            score = float(sims[idx])
            results.append(RetrievedDocument(doc=doc, score=score, rank=rank + 1))
            
        return results

class OpenAIGenerator(Generator):
    def __init__(self, client: OpenAIClient):
        self.client = client

    def generate(self, query: str, context_docs: List[Document]) -> Response:
        context_text = "\n\n".join([f"Doc {i+1}: {d.text}" for i, d in enumerate(context_docs)])
        
        prompt = (
            "Answer the question based on the context provided below.\n"
            "If the answer is not in the context, say 'I cannot answer this based on the context.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        answer_text = self.client.generate(prompt).strip()
        return Response(text=answer_text)

class SimpleRAGSystem(RAGSystem):
    def __init__(self, retriever: Retriever, generator: Generator):
        self._retriever = retriever
        self._generator = generator

    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        retrieved = self._retriever.retrieve(sample.query, top_k=top_k)
        docs = [r.doc for r in retrieved]
        response = self._generator.generate(sample.query, docs)
        return SystemOutputs(retrieved=retrieved, response=response)

def build_rag_system() -> RAGSystem:
    """
    Builds a real TF-IDF + OpenAI RAG system.
    Loads the Epstein corpus, chunks it, and indexes it.
    """
    logger.info("Initializing Simple RAG System...")
    
    # 1. Load Corpus
    # We assume the standard Epstein dataset is used.
    # In a production setting, this config should be parameterized.
    corpus_config = HFCorpusConfig(
        dataset_name="tensonaut/EPSTEIN_FILES_20K",
        split="train",
        text_column="text",
        id_column="filename",
        limit=2000 # Limit to ensure reasonable startup time for this demo
    )
    
    logger.info(f"Loading corpus from {corpus_config.dataset_name} (limit={corpus_config.limit})...")
    adapter = HFCorpusAdapter(corpus_config)
    raw_docs = adapter.to_documents()
    
    # 2. Chunking
    # Must match the generation config to align with ground truth chunks if possible.
    # config.tagging_max_output_tokens is 512 by default.
    chunk_config = ChunkingConfig(
        max_tokens=config.tagging_max_output_tokens,
        overlap_tokens=50
    )
    logger.info("Chunking documents...")
    chunks = chunk_documents(raw_docs, chunk_config)
    
    # 3. Build Retriever
    retriever = TfidfRetriever(chunks)
    
    # 4. Build Generator
    if config.openai_api_key:
        client = OpenAIClient(
            api_key=config.openai_api_key,
            model=config.tagging_model_name
        )
        generator = OpenAIGenerator(client)
    else:
        logger.warning("OPENAI_API_KEY not found. Using dummy generator.")
        # Dummy generator
        class DummyGenerator(Generator):
            def generate(self, query: str, context_docs: List[Document]) -> Response:
                return Response(text="OpenAI API Key missing. Cannot generate answer.")
        generator = DummyGenerator()

    return SimpleRAGSystem(retriever, generator)

