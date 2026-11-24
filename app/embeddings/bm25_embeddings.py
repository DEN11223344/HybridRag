import re
import math
import hashlib
from typing import List, Dict, Any, Tuple
from collections import Counter

class BM25SparseEmbedder:
    """
    Qdrant-compatible BM25 Sparse Vector Generator.
    Maps document and query terms to (indices, values) format matching Qdrant SparseVector.
    Handles exact terms, technical jargon, numbers, abbreviations, and named entities.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75, avg_doc_len: float = 250.0):
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len
        self.vocab_size = 2**30 - 1  # 30-bit positive hash space for Qdrant sparse index

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text preserving numbers, hyphenated terms, and alphanumeric acronyms.
        """
        if not text:
            return []
        # Matches words, acronyms (e.g. RAG, API-v2), decimals, integers
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return tokens

    def _term_to_index(self, term: str) -> int:
        """
        Maps a term deterministically to a positive 32-bit integer index.
        """
        h = int(hashlib.md5(term.encode("utf-8")).hexdigest()[:8], 16)
        return h % self.vocab_size + 1  # 1-indexed positive integers

    def encode_document(self, text: str) -> Dict[str, Any]:
        """
        Encodes a document chunk into a Qdrant SparseVector representation:
        { "indices": [...], "values": [...] }
        Uses BM25 term frequency saturation:
        tf_weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))
        """
        tokens = self.tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        doc_len = len(tokens)
        counts = Counter(tokens)

        indices = []
        values = []

        # Sort by term index for Qdrant sparse vector indexing efficiency
        sorted_terms = sorted(counts.items(), key=lambda x: self._term_to_index(x[0]))

        for term, tf in sorted_terms:
            idx = self._term_to_index(term)
            # BM25 TF weight
            norm_factor = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
            weight = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm_factor)
            indices.append(idx)
            values.append(round(float(weight), 4))

        return {"indices": indices, "values": values}

    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Encodes a search query into a Qdrant SparseVector representation.
        Query terms are weighted with standard query-side frequency weighting.
        """
        tokens = self.tokenize(query)
        if not tokens:
            return {"indices": [], "values": []}

        counts = Counter(tokens)
        sorted_terms = sorted(counts.items(), key=lambda x: self._term_to_index(x[0]))

        indices = []
        values = []

        for term, qtf in sorted_terms:
            idx = self._term_to_index(term)
            # Query weight
            q_weight = 1.0 + math.log(1.0 + qtf)
            indices.append(idx)
            values.append(round(float(q_weight), 4))

        return {"indices": indices, "values": values}

    def encode_documents_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.encode_document(t) for t in texts]
