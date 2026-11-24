from typing import List, Optional
import numpy as np

class Embedder:
    """Generates embeddings using sentence-transformers or mock vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                # If sentence-transformers or PyTorch is unavailable, fall back to simple hashing embedder
                self._model = "fallback"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        self._load_model()
        if self._model == "fallback":
            return [self._fallback_embed(t) for t in texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """Deterministic hash-based vector for fallback/testing without neural weights."""
        vec = np.zeros(dim, dtype=np.float32)
        for i, word in enumerate(text.lower().split()):
            h = hash(word) % dim
            vec[h] += 1.0 / (1.0 + (i * 0.1))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
