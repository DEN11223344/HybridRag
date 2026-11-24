import os
import math
import random
import logging
from typing import List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

class DenseEmbedder:
    """
    Dense Semantic Embedder using BAAI/bge-small-en-v1.5.
    Produces 384-dimensional unit-normalized embeddings on CPU.
    """
    _instance: Optional["DenseEmbedder"] = None
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384, batch_size: int = 16):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._use_fallback = False

    @classmethod
    def get_instance(cls, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384, batch_size: int = 16) -> "DenseEmbedder":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, dim=dim, batch_size=batch_size)
        return cls._instance

    def _load_model(self):
        if self._is_loaded:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Ensure CPU execution
            torch.set_num_threads(int(os.getenv("CPU_NUM_THREADS", "4")))
            logger.info(f"Loading dense embedding model on CPU: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.eval()
            self._is_loaded = True
            logger.info("Dense embedding model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Dense model '{self.model_name}' on CPU unavailable ({e}). "
                "Enabling deterministic fallback dense embedder."
            )
            self._use_fallback = True
            self._is_loaded = True

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates 384-dimensional normalized embeddings for a batch of passage texts on CPU.
        """
        if not texts:
            return []

        self._load_model()

        if self._use_fallback:
            return [self._fallback_embed(t) for t in texts]

        try:
            import torch
            all_embeddings = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self._model(**encoded)
                    # BGE uses CLS token pooling: outputs[0][:, 0]
                    cls_embeddings = outputs[0][:, 0]
                    # L2 Normalization for cosine similarity
                    normalized_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
                    batch_vecs = normalized_embeddings.cpu().numpy().tolist()
                    all_embeddings.extend(batch_vecs)

            return all_embeddings
        except Exception as e:
            logger.error(f"Error during dense inference: {e}. Falling back to deterministic embedder.")
            return [self._fallback_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string. For BGE models, adds query instruction prefix.
        """
        # BGE models recommend prepending instruction for retrieval queries
        formatted_query = f"{self.BGE_QUERY_PREFIX}{query}" if "bge" in self.model_name.lower() else query
        res = self.embed_texts([formatted_query])
        return res[0] if res else [0.0] * self.dim

    def _fallback_embed(self, text: str) -> List[float]:
        """
        Deterministic, seed-based unit-normalized pseudo embedding of size self.dim.
        """
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed_val = int.from_bytes(h[:4], "little")

        if HAS_NUMPY:
            np.random.seed(seed_val)
            vec = np.random.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        else:
            rng = random.Random(seed_val)
            raw = [rng.gauss(0, 1) for _ in range(self.dim)]
            norm = math.sqrt(sum(x * x for x in raw))
            if norm > 0:
                return [round(x / norm, 6) for x in raw]
            return [0.0] * self.dim
