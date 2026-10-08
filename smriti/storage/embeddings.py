"""
embeddings.py - In-process CPU vector embeddings using FastEmbed and NumPy.
Provides dense vector representations and cosine similarity for semantic memory recall
and cortical rule deduplication without external server dependencies.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    _instance: Optional["EmbeddingEngine"] = None
    _model = None

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._initialized = False

    def _ensure_model(self):
        """Lazy loads the FastEmbed model."""
        if self._initialized:
            return
        try:
            from fastembed import TextEmbedding
            # Suppress excessive fastembed / onnx logging if needed
            self._model = TextEmbedding(model_name=self.model_name)
            self._initialized = True
            logger.info("FastEmbed embedding engine loaded: %s", self.model_name)
        except Exception as e:
            logger.warning(
                "Failed to initialize FastEmbed model '%s': %s. Falling back to deterministic lexical vector.",
                self.model_name,
                e
            )
            self._model = None
            self._initialized = True

    def embed_text(self, text: str) -> list[float]:
        """Embeds a single string into a 384-dimensional vector."""
        if not text or not text.strip():
            return [0.0] * 384
        
        self._ensure_model()
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text]))
                if embeddings:
                    return embeddings[0].tolist()
            except Exception as e:
                logger.error("Error generating FastEmbed embedding: %s", e)

        # Fallback deterministic lexical hash vector for offline/test reliability
        return self._fallback_hash_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of strings into dense vectors."""
        if not texts:
            return []
        
        self._ensure_model()
        if self._model is not None:
            try:
                raw_embeds = list(self._model.embed(texts))
                return [vec.tolist() for vec in raw_embeds]
            except Exception as e:
                logger.error("Error generating FastEmbed batch embeddings: %s", e)

        return [self._fallback_hash_embedding(t) for t in texts]

    @staticmethod
    def _fallback_hash_embedding(text: str, dim: int = 384) -> list[float]:
        """
        Deterministic, n-gram lexical feature projection vector.
        Guarantees that identical/near-identical strings have high cosine similarity
        even if FastEmbed ONNX weights are not loaded.
        """
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.lower().strip().split()
        if not tokens:
            return vec.tolist()

        for token in tokens:
            # Hash token and character tri-grams
            h = hash(token) % dim
            vec[h] += 1.0
            for i in range(len(token) - 2):
                ngram_h = hash(token[i:i+3]) % dim
                vec[ngram_h] += 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Computes cosine similarity between two float vectors in [-1.0, 1.0]."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        
        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(-1.0, min(1.0, sim))

_global_embedding_engine: EmbeddingEngine | None = None

def get_embedding_engine(model_name: str | None = None) -> EmbeddingEngine:
    global _global_embedding_engine
    if _global_embedding_engine is None or (model_name and _global_embedding_engine.model_name != model_name):
        _global_embedding_engine = EmbeddingEngine(model_name or "BAAI/bge-small-en-v1.5")
    return _global_embedding_engine
