"""Semantic embedding service – converts text to vectors using sentence-transformers."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model '%s' …", _MODEL_NAME)
            _model = SentenceTransformer(_MODEL_NAME)
            logger.info("Model '%s' loaded successfully.", _MODEL_NAME)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed – semantic search disabled. "
                "Install with: pip install sentence-transformers"
            )
            _model = None
        except Exception:
            logger.exception("Failed to load sentence-transformer model")
            _model = None
    return _model


def encode_text(text: str) -> Optional[list[float]]:
    """Encode a single text string into a vector embedding."""
    model = _get_model()
    if model is None:
        return None
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def encode_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """Encode multiple text strings into vector embeddings (batched)."""
    model = _get_model()
    if model is None:
        return None
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_item_text(item: dict) -> str:
    """Build a single text representation of an item for embedding."""
    parts = [
        item.get("title", ""),
        item.get("description", ""),
        item.get("category", ""),
        " ".join(item.get("tags", [])),
    ]
    return " ".join(p for p in parts if p)
