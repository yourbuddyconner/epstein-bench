from typing import List, Optional, Dict
import logging

from ..core.types import Document, EvaluationSample
from ..core.dataset import InMemoryDataset

from .config import TaskGeneratorConfig, LLMClient
from .hf_adapter import HFCorpusAdapter
from .preprocess import chunk_documents, embed_chunks
from .anchors import extract_anchors, Anchor
from .single_hop import generate_single_hop_tasks
from .multi_hop import generate_multi_hop_tasks
from .robustness import generate_paraphrase_variants, generate_unanswerable_tasks
from .filters import (
    apply_filters,
    SampleFilter,
    AnswerPresenceFilter,
    QuestionLengthFilter,
    LLMConsistencyFilter,
)
from .retriever import CorpusRetriever

logger = logging.getLogger(__name__)


def group_anchors_by_doc(anchors: List[Anchor]) -> Dict[str, List[Anchor]]:
    out = {}
    for a in anchors:
        out.setdefault(a.doc_id, []).append(a)
    return out


class TaskGenerator:
    def __init__(
        self,
        llm: LLMClient,
        config: TaskGeneratorConfig,
        sample_filters: Optional[List[SampleFilter]] = None,
    ):
        self.llm = llm
        self.config = config
        self.sample_filters = sample_filters or []
        self.last_entity_graph: Dict[str, List[str]] = {}

    def build_corpus(self) -> List[Document]:
        if self.config.hf_corpus is not None:
            logger.info(f"Building corpus from HF config: {self.config.hf_corpus.dataset_name}")
            adapter = HFCorpusAdapter(self.config.hf_corpus)
            docs = adapter.to_documents()
            logger.info(f"Loaded {len(docs)} documents from corpus.")
            return docs
        else:
            raise ValueError("TaskGenerator requires an HFCorpusConfig or another corpus source.")

    def generate_samples(self) -> List[EvaluationSample]:
        # 1) Build corpus
        logger.info("Step 1/7: Building Corpus...")
        docs = self.build_corpus()

        # 2) Preprocess: chunk + embeddings
        logger.info(f"Step 2/7: Preprocessing (Chunking: max_tokens={self.config.chunking.max_tokens})...")
        chunks = chunk_documents(docs, self.config.chunking)
        logger.info(f"Generated {len(chunks)} chunks.")
        
        retriever: Optional[CorpusRetriever] = None
        if self.config.embedding is not None:
            logger.info("Step 2b/7: Embedding chunks...")
            embeddings = embed_chunks(chunks, self.config.embedding)
            try:
                retriever = CorpusRetriever(
                    documents=chunks,
                    embeddings=embeddings,
                    embed_fn=self.config.embedding.embed_fn,
                )
            except Exception as exc:
                logger.warning("Failed to initialise retriever: %s", exc)
                retriever = None

        # 3) Anchors
        logger.info("Step 3/7: Extracting Anchors...")
        anchors = extract_anchors(chunks)
        logger.info(f"Extracted {len(anchors)} anchors.")

        # 4) Single-hop tasks
        all_samples: List[EvaluationSample] = []
        if self.config.single_hop is not None:
            logger.info("Step 4/7: Generating Single-Hop Tasks...")
            single_hop_samples = generate_single_hop_tasks(
                anchors=anchors,
                chunks=chunks,
                llm=self.llm,
                cfg=self.config.single_hop,
                max_workers=self.config.max_workers,
            )
            logger.info(f"Generated {len(single_hop_samples)} single-hop samples.")
            all_samples.extend(single_hop_samples)

        # 5) Multi-hop tasks (optional)
        if self.config.multi_hop is not None:
            logger.info("Step 5/7: Generating Multi-Hop Tasks...")
            anchors_by_doc = group_anchors_by_doc(anchors)
            multi_hop_samples, graph = generate_multi_hop_tasks(
                chunks=chunks,
                anchors_by_doc=anchors_by_doc,
                llm=self.llm,
                cfg=self.config.multi_hop,
                max_workers=self.config.max_workers,
            )
            self.last_entity_graph = graph
            logger.info(f"Generated {len(multi_hop_samples)} multi-hop samples.")
            all_samples.extend(multi_hop_samples)

        # 6) Robustness variants (optional)
        if self.config.robustness is not None:
            logger.info("Step 6/7: Generating Robustness Variants...")
            paraphrases = generate_paraphrase_variants(
                base_samples=all_samples,
                llm=self.llm,
                cfg=self.config.robustness,
                max_workers=self.config.max_workers,
            )
            logger.info(f"Generated {len(paraphrases)} paraphrase variants.")
            all_samples.extend(paraphrases)

            negatives = generate_unanswerable_tasks(
                chunks=chunks,
                llm=self.llm,
                cfg=self.config.robustness,
                retriever=retriever,
                max_workers=self.config.max_workers,
            )
            logger.info(f"Generated {len(negatives)} unanswerable tasks.")
            all_samples.extend(negatives)

        # 7) Filters
        filters = self.sample_filters
        if not filters:
            filters = [
                AnswerPresenceFilter(exclude_types={"multi_hop"}),
                QuestionLengthFilter(),
                LLMConsistencyFilter(llm=self.llm, include_types={"multi_hop"}),
            ]

        if filters:
            logger.info("Step 7/7: Applying Filters...")
            initial_count = len(all_samples)
            all_samples = apply_filters(all_samples, filters, max_workers=self.config.max_workers)
            logger.info(f"Filtered out {initial_count - len(all_samples)} samples. Remaining: {len(all_samples)}")

        return all_samples

    def generate_dataset(self, name: str = "generated") -> InMemoryDataset:
        samples = self.generate_samples()
        return InMemoryDataset(name=name, samples=samples)
