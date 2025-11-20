"""
Baseline RAG implementation for Epstein Bench.
Uses simple TF-IDF retrieval and extractive answering.
"""

import json
import time
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch


class BaselineTFIDFRAG:
    """
    Simple baseline RAG using TF-IDF for retrieval and extractive answering.
    This serves as a minimum baseline for the benchmark.
    """
    
    def __init__(self, documents_file: str, top_k: int = 10):
        """
        Args:
            documents_file: Path to processed documents JSON
            top_k: Number of documents to retrieve
        """
        print("Loading documents...")
        with open(documents_file, 'r') as f:
            self.documents = json.load(f)
        
        self.top_k = top_k
        self.doc_texts = [doc['text'] for doc in self.documents]
        self.doc_ids = [doc['doc_id'] for doc in self.documents]
        
        # Build TF-IDF index
        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.doc_vectors = self.vectorizer.fit_transform(self.doc_texts)
        print(f"Indexed {len(self.documents)} documents")
    
    def retrieve_and_answer(self, question: str) -> Tuple[str, List[str], Dict]:
        """
        Retrieve documents and generate answer for a question.
        
        Returns:
            - answer: Extracted answer
            - retrieved_doc_ids: List of retrieved document IDs
            - timing: Timing information
        """
        timing = {}
        
        # Step 1: Retrieve documents
        retrieval_start = time.time()
        retrieved_docs, retrieved_ids = self._retrieve_documents(question)
        timing['retrieval_ms'] = (time.time() - retrieval_start) * 1000
        
        # Step 2: Extract answer
        generation_start = time.time()
        answer = self._extract_answer(question, retrieved_docs)
        timing['generation_ms'] = (time.time() - generation_start) * 1000
        
        return answer, retrieved_ids, timing
    
    def _retrieve_documents(self, question: str) -> Tuple[List[str], List[str]]:
        """Retrieve top-k documents using TF-IDF similarity."""
        # Vectorize the question
        question_vec = self.vectorizer.transform([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vec, self.doc_vectors).flatten()
        
        # Get top-k documents
        top_indices = similarities.argsort()[-self.top_k:][::-1]
        
        retrieved_texts = [self.doc_texts[i] for i in top_indices]
        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        
        return retrieved_texts, retrieved_ids
    
    def _extract_answer(self, question: str, documents: List[str]) -> str:
        """Extract answer from retrieved documents using simple heuristics."""
        # Extract question type and key term
        key_term = self._extract_key_term(question)
        
        if not key_term:
            # If no key term found, return first sentence from top document
            if documents and documents[0]:
                sentences = documents[0].split('.')
                return sentences[0].strip() + '.' if sentences else "No answer found."
            return "No answer found."
        
        # Search for key term in documents and extract surrounding context
        for doc in documents:
            if key_term.lower() in doc.lower():
                # Find the sentence containing the key term
                sentences = doc.split('.')
                for sentence in sentences:
                    if key_term.lower() in sentence.lower():
                        return sentence.strip() + '.'
        
        # Fallback: return first relevant sentence
        for doc in documents[:3]:
            sentences = doc.split('.')
            if sentences:
                return sentences[0].strip() + '.'
        
        return "No answer found."
    
    def _extract_key_term(self, question: str) -> Optional[str]:
        """Extract the key term (needle) from the question."""
        # Try to extract dates
        date_match = re.search(
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            question,
            re.IGNORECASE
        )
        if date_match:
            return date_match.group(0)
        
        # Try to extract emails
        email_match = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', question)
        if email_match:
            return email_match.group(0)
        
        # Try to extract phone numbers
        phone_match = re.search(r'(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}', question)
        if phone_match:
            return phone_match.group(0)
        
        # Try to extract money amounts
        money_match = re.search(r'\$[\d,]+(?:\.\d{2})?', question)
        if money_match:
            return money_match.group(0)
        
        # Try to extract flight numbers
        flight_match = re.search(r'(?:flight|FLT|FL)\s*#?\s*([A-Z]{2,3}\s*\d{2,4})', question, re.IGNORECASE)
        if flight_match:
            return flight_match.group(0)
        
        # Try to extract addresses
        address_match = re.search(
            r'\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)',
            question
        )
        if address_match:
            return address_match.group(0)
        
        return None


class EmbeddingRAG:
    """
    More advanced RAG using sentence embeddings for retrieval.
    Better semantic understanding than TF-IDF.
    """
    
    def __init__(self, documents_file: str, top_k: int = 10, 
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Args:
            documents_file: Path to processed documents JSON
            top_k: Number of documents to retrieve
            model_name: Sentence transformer model to use
        """
        print("Loading documents...")
        with open(documents_file, 'r') as f:
            self.documents = json.load(f)
        
        self.top_k = top_k
        self.doc_texts = [doc['text'][:2000] for doc in self.documents]  # Truncate for efficiency
        self.doc_ids = [doc['doc_id'] for doc in self.documents]
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)
        
        # Build document embeddings
        print("Building document embeddings (this may take a while)...")
        batch_size = 32
        self.doc_embeddings = []
        
        for i in range(0, len(self.doc_texts), batch_size):
            batch = self.doc_texts[i:i+batch_size]
            batch_embeddings = self.embed_model.encode(batch, show_progress_bar=False)
            self.doc_embeddings.extend(batch_embeddings)
        
        self.doc_embeddings = np.array(self.doc_embeddings)
        print(f"Built embeddings for {len(self.documents)} documents")
    
    def retrieve_and_answer(self, question: str) -> Tuple[str, List[str], Dict]:
        """
        Retrieve documents and generate answer for a question.
        """
        timing = {}
        
        # Step 1: Retrieve documents
        retrieval_start = time.time()
        retrieved_docs, retrieved_ids = self._retrieve_documents(question)
        timing['retrieval_ms'] = (time.time() - retrieval_start) * 1000
        
        # Step 2: Extract answer (reuse from baseline)
        generation_start = time.time()
        answer = self._extract_answer(question, retrieved_docs)
        timing['generation_ms'] = (time.time() - generation_start) * 1000
        
        return answer, retrieved_ids, timing
    
    def _retrieve_documents(self, question: str) -> Tuple[List[str], List[str]]:
        """Retrieve top-k documents using embedding similarity."""
        # Encode question
        question_embedding = self.embed_model.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, self.doc_embeddings).flatten()
        
        # Get top-k documents
        top_indices = similarities.argsort()[-self.top_k:][::-1]
        
        # Return full texts (not truncated)
        retrieved_texts = [self.documents[i]['text'] for i in top_indices]
        retrieved_ids = [self.doc_ids[i] for i in top_indices]
        
        return retrieved_texts, retrieved_ids
    
    def _extract_answer(self, question: str, documents: List[str]) -> str:
        """Extract answer using the same method as baseline."""
        # Reuse the baseline extraction logic
        baseline = BaselineTFIDFRAG.__new__(BaselineTFIDFRAG)
        return baseline._extract_answer(baseline, question, documents)
    
    def _extract_key_term(self, question: str) -> Optional[str]:
        """Extract key term using baseline method."""
        baseline = BaselineTFIDFRAG.__new__(BaselineTFIDFRAG)
        return baseline._extract_key_term(baseline, question)


def test_baseline():
    """Test the baseline RAG systems."""
    # Make sure documents are available
    docs_file = "./data/processed_documents.json"
    if not Path(docs_file).exists():
        print("Please run generate_benchmark.py first to create processed documents")
        return
    
    # Initialize baseline RAG
    print("\n=== Testing TF-IDF Baseline ===")
    tfidf_rag = BaselineTFIDFRAG(docs_file, top_k=10)
    
    # Test questions
    test_questions = [
        "What happened on December 15, 2019?",
        "Who is associated with the email john.doe@example.com?",
        "What information is connected to flight AA123?"
    ]
    
    for q in test_questions:
        answer, docs, timing = tfidf_rag.retrieve_and_answer(q)
        print(f"\nQuestion: {q}")
        print(f"Answer: {answer}")
        print(f"Retrieved {len(docs)} documents in {timing['retrieval_ms']:.1f}ms")
        print(f"Generated answer in {timing['generation_ms']:.1f}ms")
    
    # Optionally test embedding RAG (requires more memory)
    # print("\n=== Testing Embedding Baseline ===")
    # embed_rag = EmbeddingRAG(docs_file, top_k=10)
    # for q in test_questions:
    #     answer, docs, timing = embed_rag.retrieve_and_answer(q)
    #     print(f"\nQuestion: {q}")
    #     print(f"Answer: {answer[:100]}...")


if __name__ == "__main__":
    test_baseline()
