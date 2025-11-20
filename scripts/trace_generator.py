"""
TRACe Benchmark Generator for Epstein Bench.
Creates challenging single-fact retrieval questions from the dataset.
Uses OpenAI to generate high-quality questions and answers aligned with TRACe metrics.
"""

import json
import random
import re
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()


@dataclass
class TraceQuestion:
    """Represents a single TRACe benchmark question."""
    question_id: str
    question: str
    answer: str
    source_doc_id: str
    source_filename: str
    context: str
    difficulty: str  # easy, medium, hard
    question_type: str  # date, name, location, number, etc.
    needle_rarity: float  # How rare is this fact in the corpus
    answer_position: int  # Character position in document
    created_at: str
    selection_score: float = 0.0
    document_tags: List[str] = field(default_factory=list)
    reasoning_type: str = "fact_retrieval" # fact_retrieval, multi_hop, summary


class TraceBenchmarkGenerator:
    """Generates TRACe benchmark questions from the dataset."""
    
    # Patterns for extracting different types of facts
    PATTERNS = {
        'date': {
            'regex': r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            'type': 'date'
        },
        'phone': {
            'regex': r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',
            'type': 'phone'
        },
        'email': {
            'regex': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'type': 'email'
        },
        'flight_number': {
            'regex': r'\b(?:Flight|FLT|FL)\s*#?\s*([A-Z]{2,3}\s*\d{2,4})\b',
            'type': 'flight'
        },
        'money': {
            'regex': r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|K|M|B))?\b',
            'type': 'money'
        },
        'address': {
            'regex': r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Plaza|Place|Pl)\b',
            'type': 'address'
        },
        'tail_number': {
            'regex': r'\b[N][0-9]{1,5}[A-Z]{0,2}\b',
            'type': 'tail_number'
        }
    }
    
    def __init__(self, documents: List[Dict], output_dir: str = "./benchmarks", 
                 use_openai: bool = True,
                 corpus_documents: Optional[List[Dict]] = None,
                 doc_annotations: Optional[Dict[str, Any]] = None,
                 selection_metadata: Optional[Dict[str, Any]] = None,
                 model_name: str = "gpt-4o"):
        self.documents = documents
        self.corpus_documents = corpus_documents or documents
        self.doc_annotations = doc_annotations or {}
        self.selection_metadata = selection_metadata or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions: List[TraceQuestion] = []
        self.fact_frequency: Dict[Tuple[str, str], int] = {}
        self.use_openai = use_openai
        self.client: Optional[OpenAI] = None
        self.model_name = model_name
        
        # Initialize OpenAI client if API key is available
        if use_openai:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not set. Falling back to simple extraction (not recommended).")
                self.use_openai = False
            else:
                self.client = OpenAI(api_key=api_key)
                print(f"OpenAI client initialized for QA generation using {self.model_name}")
                
        # Prepare TF-IDF for hard negatives (only on corpus if needed)
        # We lazy load this only if we are adding distractors later to save time
        self.vectorizer = None
        self.doc_vectors = None
        self.doc_ids_map = []
    
    def _is_high_quality_context(self, text: str) -> bool:
        """Check if context is high quality (low OCR noise)."""
        if not text:
            return False
            
        # Heuristic 1: Alpha-numeric ratio
        # Calculate ratio of alphanumeric chars to total chars (excluding spaces)
        clean_chars = len([c for c in text if c.isalnum()])
        total_chars = len([c for c in text if not c.isspace()])
        
        if total_chars == 0:
            return False
            
        ratio = clean_chars / total_chars
        if ratio < 0.7:  # Threshold for garbage/OCR noise
            return False
            
        # Heuristic 2: Average word length (too short = noise, too long = noise)
        words = text.split()
        if not words:
            return False
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 2 or avg_word_len > 15:
            return False
            
        return True

    def generate_trace_questions(self, 
                              num_questions: int = 100,
                              min_context_length: int = 150,
                              max_context_length: int = 1000) -> List[TraceQuestion]:
        """Generate TRACe questions by finding rare facts in documents."""
        print(f"Generating {num_questions} TRACe questions...")
        
        # First pass: collect all facts and their frequencies
        self._collect_fact_frequencies()
        
        # Second pass: generate questions from rare facts
        all_candidates = []
        
        for doc in tqdm(self.documents, desc="Extracting candidates"):
            if not doc['text'] or len(doc['text']) < min_context_length:
                continue
                
            # Filter for OCR quality FIRST
            if not self._is_high_quality_context(doc['text'][:max_context_length]): # Check start of doc as proxy
                 continue
            
            candidates = self._extract_fact_candidates(doc)
            all_candidates.extend(candidates)
        
        # Sort by rarity (lower frequency = rarer)
        all_candidates.sort(key=lambda x: x['frequency'])
        
        # Select the rarest facts for questions, ensuring diversity
        selected_candidates = []
        used_facts = set()
        used_docs = set()
        
        # Strategy: Prioritize 1 candidate per doc, then fill up
        for candidate in all_candidates:
            fact_key = (candidate['type'], candidate['value'])
            
            # Skip if we've already used this exact fact
            if fact_key in used_facts:
                continue
                
            # Skip if we already have enough questions
            if len(selected_candidates) >= num_questions * 2.0:  # Over-sample more for failures
                break
                
            selected_candidates.append(candidate)
            used_facts.add(fact_key)
        
        # Limit to requested number after generation (since some generations might fail)
        print(f"Selected {len(selected_candidates)} candidates for generation. Processing...")
        
        # Convert candidates to questions
        for candidate in tqdm(selected_candidates):
            if len(self.questions) >= num_questions:
                break
                
            question = self._create_question(candidate, min_context_length, max_context_length)
            if question:
                self.questions.append(question)
        
        print(f"Successfully generated {len(self.questions)} TRACe questions")
        return self.questions
    
    def _collect_fact_frequencies(self) -> None:
        """Collect frequency of each fact across the corpus."""
        print("Collecting fact frequencies...")
        
        for doc in tqdm(self.documents):
            if not doc['text']:
                continue
            
            for pattern_name, pattern_info in self.PATTERNS.items():
                matches = re.finditer(pattern_info['regex'], doc['text'], re.IGNORECASE)
                for match in matches:
                    fact_value = match.group(0).strip()
                    fact_key = (pattern_info['type'], fact_value.lower())
                    
                    if fact_key not in self.fact_frequency:
                        self.fact_frequency[fact_key] = 0
                    self.fact_frequency[fact_key] += 1
    
    def _extract_fact_candidates(self, doc: Dict) -> List[Dict]:
        """Extract potential TRACe facts from a document."""
        candidates = []
        
        for pattern_name, pattern_info in self.PATTERNS.items():
            matches = re.finditer(pattern_info['regex'], doc['text'], re.IGNORECASE)
            
            for match in matches:
                fact_value = match.group(0).strip()
                fact_key = (pattern_info['type'], fact_value.lower())
                frequency = self.fact_frequency.get(fact_key, 0)
                
                # Only consider rare facts (appear in fewer than 5 documents)
                if frequency > 0 and frequency <= 5:
                    candidates.append({
                        'doc_id': doc['doc_id'],
                        'filename': doc['filename'],
                        'type': pattern_info['type'],
                        'pattern_name': pattern_name,
                        'value': fact_value,
                        'position': match.start(),
                        'frequency': frequency,
                        'text': doc['text']
                    })
        
        return candidates
    
    def _generate_qa_pair(self, context: str, fact_value: str, fact_type: str) -> Tuple[str, str, str]:
        """Generate a natural language Question and Answer pair using OpenAI."""
        if not self.use_openai or not self.client:
            return None, None, "fact_retrieval"
        
        try:
            prompt = f"""You are an expert legal analyst creating a benchmark dataset for RAG evaluation (TRACe).
Your task is to create a CHALLENGING reasoning question based on the provided text segment.

Context Segment:
{context}

Key Fact to Pivot On: "{fact_value}" (Type: {fact_type})

Instructions:
1. Create a question that requires **combining two pieces of information** from this segment or **summarizing a key event/implication** related to "{fact_value}".
   - AVOID simple lookup questions like "What is the date?" or "What is the email?".
   - Instead, ask "What was the outcome of the meeting scheduled for {fact_value}?" or "Who sent the email regarding the merger to {fact_value}?".
2. Provide a **Complete Sentence Answer** that explicitly cites the context.
   - Format: "The document states that [Answer]..." or "According to the text, [Answer]..."
   - The answer MUST be fully supported by the provided Context Segment.
   - Do not hallucinate information not present in the segment.

Output format (JSON):
{{
  "question": "The generated question string",
  "answer": "The complete sentence answer string",
  "reasoning_type": "multi_hop" or "summary" or "fact_retrieval"
}}"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant creating benchmark datasets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("question"), data.get("answer"), data.get("reasoning_type", "fact_retrieval")
            
        except Exception as e:
            print(f"Error generating QA pair: {e}")
            return None, None, "fact_retrieval"
    
    def _create_question(self, candidate: Dict, min_context: int, max_context: int) -> Optional[TraceQuestion]:
        """Create a TRACe question from a fact candidate."""
        try:
            # Extract context around the fact
            text = candidate['text']
            pos = candidate['position']
            
            # Find sentence boundaries for context window
            # We want a decent chunk of text to make it challenging (Needle in Haystack)
            start = max(0, pos - max_context // 2)
            end = min(len(text), pos + max_context // 2)
            
            # Align to nearest sentence boundaries
            context_start = text.rfind('.', max(0, start - 100), start)
            if context_start == -1: context_start = start
            else: context_start += 1
            
            context_end = text.find('.', end, min(len(text), end + 100))
            if context_end == -1: context_end = end
            else: context_end += 1
            
            context = text[context_start:context_end].strip()
            
            # Double check quality of the extracted window
            if not self._is_high_quality_context(context):
                return None

            # Validate context contains the needle
            if candidate['value'] not in context:
                # If strict containment fails (due to boundary issues), try expanding or fallback
                return None
            
            if len(context) < min_context:
                return None
            
            # Generate Q&A using LLM
            question_text, answer_text, reasoning_type = self._generate_qa_pair(context, candidate['value'], candidate['type'])
            
            if not question_text or not answer_text:
                return None
            
            # Determine difficulty
            if candidate['frequency'] == 1:
                difficulty = 'hard'
            elif candidate['frequency'] <= 3:
                difficulty = 'medium'
            else:
                difficulty = 'easy'
            
            # Generate ID
            question_id = hashlib.md5(
                f"{candidate['doc_id']}_{candidate['value']}_{candidate['position']}".encode()
            ).hexdigest()[:12]
            
            annotation = self.doc_annotations.get(candidate['doc_id'], {}) if hasattr(self, "doc_annotations") else {}
            selection_score = float(annotation.get('score', 0.0)) if annotation else 0.0
            document_tags = list(annotation.get('document_tags', [])) if annotation else []
            
            return TraceQuestion(
                question_id=f"trace_{question_id}",
                question=question_text,
                answer=answer_text,
                source_doc_id=candidate['doc_id'],
                source_filename=candidate['filename'],
                context=context,
                difficulty=difficulty,
                question_type=candidate['type'],
                needle_rarity=1.0 / candidate['frequency'],
                answer_position=candidate['position'],
                created_at=datetime.now().isoformat(),
                selection_score=selection_score,
                document_tags=document_tags,
                reasoning_type=reasoning_type
            )
            
        except Exception as e:
            print(f"Error processing candidate: {e}")
            return None
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for hard negative mining."""
        if self.vectorizer is not None:
            return
            
        print("Building TF-IDF index for hard negative mining...")
        corpus_texts = [d.get('text', '')[:1000] for d in self.corpus_documents] # First 1000 chars is enough for rough matching
        self.doc_ids_map = [d['doc_id'] for d in self.corpus_documents]
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(corpus_texts)
        print("TF-IDF index built.")

    def add_distractor_documents(self, num_distractors: int = 100) -> List[str]:
        """Select HARD NEGATIVE distractor documents that don't contain the answers."""
        print(f"Selecting {num_distractors} hard negative distractor documents...")
        
        self._build_tfidf_index()
        
        # Get source doc indices
        source_doc_ids = {q.source_doc_id for q in self.questions}
        
        # Find documents that are similar to the source documents but NOT the source documents
        # We'll sample a few questions and find their nearest neighbors
        
        hard_negatives = set()
        
        # Collect source text vectors
        # Ideally we'd do this per question, but for speed let's grab a random sample of source docs
        # and find their neighbors.
        
        # Better approach: For each question, find top-k similar docs to its CONTEXT
        # and add them if they are not the source doc.
        
        print("Mining hard negatives...")
        # Limit to mining from a subset of questions to save time if we have many
        sample_questions = self.questions[:min(len(self.questions), 50)] 
        
        context_vectors = self.vectorizer.transform([q.context for q in sample_questions])
        
        # Compute similarity against all docs
        similarities = cosine_similarity(context_vectors, self.doc_vectors)
        
        for i, q in enumerate(sample_questions):
            # Get top 5 similar docs
            top_indices = similarities[i].argsort()[-10:][::-1]
            
            for idx in top_indices:
                candidate_id = self.doc_ids_map[idx]
                if candidate_id != q.source_doc_id:
                    hard_negatives.add(candidate_id)
            
            if len(hard_negatives) >= num_distractors:
                break
        
        # If we don't have enough hard negatives, fill with random
        hard_negatives = list(hard_negatives)
        if len(hard_negatives) < num_distractors:
            remaining = num_distractors - len(hard_negatives)
            print(f"Need {remaining} more distractors, filling with random...")
            
            all_ids = [d['doc_id'] for d in self.corpus_documents]
            available = [d for d in all_ids if d not in source_doc_ids and d not in hard_negatives]
            
            if available:
                random_fill = random.sample(available, min(len(available), remaining))
                hard_negatives.extend(random_fill)
        
        return hard_negatives[:num_distractors]
    
    def save_benchmark(self, name: str = "trace_benchmark", num_distractors: Optional[int] = None) -> None:
        """Save the TRACe benchmark to JSON files."""
        if not self.questions:
            print("No questions to save")
            return
        
        metadata = {
            'name': name,
            'version': '3.0',  # Bump version for TRACe
            'created_at': datetime.now().isoformat(),
            'num_questions': len(self.questions),
            'question_types': list(set(q.question_type for q in self.questions)),
            'difficulty_distribution': {
                'easy': sum(1 for q in self.questions if q.difficulty == 'easy'),
                'medium': sum(1 for q in self.questions if q.difficulty == 'medium'),
                'hard': sum(1 for q in self.questions if q.difficulty == 'hard')
            },
            'generator_model': self.model_name,
            'methodology': 'trace_reasoning'
        }
        
        questions_data = [asdict(q) for q in self.questions]
        
        distractor_target = num_distractors if num_distractors is not None else 100
        distractors = self.add_distractor_documents(distractor_target)
        
        benchmark = {
            'metadata': metadata,
            'questions': questions_data,
            'distractor_doc_ids': distractors
        }
        
        output_file = self.output_dir / f"{name}.json"
        with open(output_file, 'w') as f:
            json.dump(benchmark, f, indent=2)
        
        print(f"\nSaved benchmark to {output_file}")
        
        # Simplified version
        simple_file = self.output_dir / f"{name}_simple.jsonl"
        with open(simple_file, 'w') as f:
            for q in self.questions:
                simple = {
                    'id': q.question_id,
                    'question': q.question,
                    'answer': q.answer,
                    'source_doc': q.source_doc_id,
                    'context': q.context 
                }
                f.write(json.dumps(simple) + '\n')
    
    def generate_statistics(self) -> pd.DataFrame:
        """Generate statistics about the benchmark questions."""
        if not self.questions:
            return pd.DataFrame()
        
        df = pd.DataFrame([asdict(q) for q in self.questions])
        
        stats = {
            'Total Questions': len(df),
            'Avg Context Length': df['context'].str.len().mean(),
            'Avg Answer Length': df['answer'].str.len().mean(),
        }
        
        print("\n=== Benchmark Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
            
        print("\n=== Question Type Distribution ===")
        print(df['question_type'].value_counts())
        
        if 'reasoning_type' in df:
             print("\n=== Reasoning Type Distribution ===")
             print(df['reasoning_type'].value_counts())
        
        return df


if __name__ == "__main__":
    # Load processed documents
    doc_path = Path("./data/processed_documents.json")
    if doc_path.exists():
        with open(doc_path, 'r') as f:
            documents = json.load(f)
        
        # Generate TRACe benchmark
        generator = TraceBenchmarkGenerator(documents)
        questions = generator.generate_trace_questions(num_questions=50) # Smaller default for testing
        generator.save_benchmark("epstein_trace_v1")
        stats_df = generator.generate_statistics()
    else:
        print(f"Error: {doc_path} not found. Please run data processing first.")
