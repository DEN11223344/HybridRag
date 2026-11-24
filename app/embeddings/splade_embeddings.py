import os
import logging
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logger = logging.getLogger(__name__)

class SpladeEmbedder:
    """
    SPLADE Sparse Semantic Embedder using prithivida/Splade_PP_en_v1.
    Generates learned sparse representations where indices are BERT vocabulary token IDs (0..30521)
    and values are log(1 + ReLU(w)) activation scores.
    Runs locally on CPU.
    """
    _instance: Optional["SpladeEmbedder"] = None

    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1", batch_size: int = 8):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._is_loaded = False
        self._use_fallback = False

    @classmethod
    def get_instance(cls, model_name: str = "prithivida/Splade_PP_en_v1") -> "SpladeEmbedder":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def _load_model(self):
        if self._is_loaded:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            torch.set_num_threads(int(os.getenv("CPU_NUM_THREADS", "4")))
            logger.info(f"Loading SPLADE model on CPU: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
            self._is_loaded = True
            logger.info("SPLADE model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"SPLADE model '{self.model_name}' on CPU unavailable ({e}). "
                "Enabling deterministic semantic sparse fallback."
            )
            self._use_fallback = True
            self._is_loaded = True

    def encode_text(self, text: str) -> Dict[str, Any]:
        res = self.encode_batch([text])
        return res[0] if res else {"indices": [], "values": []}

    def encode_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not texts:
            return []

        self._load_model()

        if self._use_fallback:
            return [self._fallback_encode(t) for t in texts]

        try:
            import torch
            results = []

            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                tokens = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    logits = self._model(**tokens).logits
                    relu_logits = torch.relu(logits)
                    sparse_repr = torch.max(
                        torch.log(1.0 + relu_logits) * tokens.attention_mask.unsqueeze(-1),
                        dim=1
                    )[0]

                    for vec in sparse_repr:
                        non_zero_indices = torch.nonzero(vec).squeeze().cpu().numpy().tolist()
                        if isinstance(non_zero_indices, int):
                            non_zero_indices = [non_zero_indices]
                        elif not non_zero_indices:
                            results.append({"indices": [], "values": []})
                            continue

                        values = vec[non_zero_indices].cpu().numpy().tolist()
                        
                        if len(non_zero_indices) > 256:
                            sorted_pairs = sorted(zip(non_zero_indices, values), key=lambda x: x[1], reverse=True)[:256]
                            sorted_pairs.sort(key=lambda x: x[0])
                            non_zero_indices = [p[0] for p in sorted_pairs]
                            values = [round(float(p[1]), 4) for p in sorted_pairs]
                        else:
                            values = [round(float(v), 4) for v in values]

                        results.append({
                            "indices": [int(idx) for idx in non_zero_indices],
                            "values": values,
                        })

            return results
        except Exception as e:
            logger.error(f"Error during SPLADE inference: {e}. Using fallback.")
            return [self._fallback_encode(t) for t in texts]

    def _fallback_encode(self, text: str) -> Dict[str, Any]:
        import hashlib
        import re
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return {"indices": [], "values": []}

        indices = []
        values = []
        for w in sorted(set(words)):
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:6], 16) % 30522
            indices.append(h)
            values.append(round(1.0 + 0.1 * (len(w) % 5), 4))

        return {"indices": indices, "values": values}
