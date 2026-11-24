import os
import re
import logging
from typing import List, Optional
from app.schemas.search import SearchResultChunk

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "of", "and", "but", "or",
    "so", "what", "which", "who", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "than", "too", "very", "can", "will", "just", "should", "now"
}

class CrossEncoderReranker:
    """
    Reranks candidate chunks using BAAI/bge-reranker-base on CPU.
    Takes (query, document_chunk) pairs, scores relevance, handles long text truncation cleanly,
    and returns top-K most relevant chunks.
    """
    _instance: Optional["CrossEncoderReranker"] = None

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._use_fallback = False

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-reranker-base") -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load_model(self):
        if self._is_loaded:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(int(os.getenv("CPU_NUM_THREADS", "4")))
            logger.info(f"Loading reranker model on CPU: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._is_loaded = True
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Cross-encoder '{self.model_name}' on CPU unavailable ({e}). "
                "Enabling deterministic lexical-semantic fallback reranker."
            )
            self._use_fallback = True
            self._is_loaded = True

    def rerank(
        self,
        query: str,
        candidates: List[SearchResultChunk],
        top_k: int = 5
    ) -> List[SearchResultChunk]:
        if not candidates:
            return []

        self._load_model()

        if self._use_fallback:
            return self._fallback_rerank(query, candidates, top_k)

        try:
            import torch
            pairs = [[query, c.chunk_text] for c in candidates]
            
            with torch.no_grad():
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                scores = self._model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()

            scored_candidates = []
            for score, chunk in zip(scores, candidates):
                reranked_chunk = SearchResultChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    section_title=getattr(chunk, "section_title", None),
                    chunk_text=chunk.chunk_text,
                    token_count=getattr(chunk, "token_count", 0),
                    content_type=getattr(chunk, "content_type", "text"),
                    table_id=getattr(chunk, "table_id", None),
                    image_references=getattr(chunk, "image_references", []),
                    score=round(float(score), 4),
                    retrieval_method=f"reranked({getattr(chunk, 'retrieval_method', 'unknown')})",
                )
                scored_candidates.append(reranked_chunk)

            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            return scored_candidates[:top_k]
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Using fallback.")
            return self._fallback_rerank(query, candidates, top_k)

    def _fallback_rerank(self, query: str, candidates: List[SearchResultChunk], top_k: int) -> List[SearchResultChunk]:
        # Filter stopwords and extract alphanumeric terms
        raw_q_words = re.findall(r"\b\w+\b", query.lower())
        q_tokens = set(w for w in raw_q_words if w not in STOPWORDS)
        if not q_tokens:
            q_tokens = set(raw_q_words)

        scored = []
        for c in candidates:
            c_words = set(re.findall(r"\b\w+\b", c.chunk_text.lower()))
            overlap = len(q_tokens.intersection(c_words))
            # Strongly prioritize substantive term match over baseline score
            relevance_score = (overlap * 2.0) + (float(getattr(c, "score", 0.0)) * 0.1)
            reranked_chunk = SearchResultChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                page_number=c.page_number,
                section_title=getattr(c, "section_title", None),
                chunk_text=c.chunk_text,
                token_count=getattr(c, "token_count", 0),
                content_type=getattr(c, "content_type", "text"),
                table_id=getattr(c, "table_id", None),
                image_references=getattr(c, "image_references", []),
                score=round(relevance_score, 4),
                retrieval_method=f"reranked({getattr(c, 'retrieval_method', 'unknown')})",
            )
            scored.append(reranked_chunk)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
