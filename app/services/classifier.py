"""
Local embedding-based theme classifier — Step 6 of the thematic classification pipeline.

Uses SentenceTransformer (all-MiniLM-L6-v2 by default) to encode statements
and approved themes, then computes cosine similarity to find the best match.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

logger = logging.getLogger("analytics_service.services.classifier")

# Module-level model cache — loaded once per worker process
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazily load the sentence transformer model (cached at module level)."""
    global _model
    if _model is None:
        model_name = settings.EMBEDDING_MODEL_NAME
        logger.info(f"Loading SentenceTransformer model '{model_name}'...")
        _model = SentenceTransformer(model_name)
        logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
    return _model


def build_theme_embeddings(approved_themes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Build embeddings for each approved theme.
    For each theme, encodes both the name and definition separately,
    then stacks them as a multi-vector representation.

    Returns: {theme_id_str: np.ndarray of shape (N, embedding_dim)}
    """
    model = _get_model()
    theme_vectors: Dict[str, np.ndarray] = {}

    for theme in approved_themes:
        theme_id = str(theme["id"])
        name = theme.get("name", "")
        definition = theme.get("definitions", "") or theme.get("definition", "") or ""

        texts_to_encode = []
        if name:
            texts_to_encode.append(f"Theme: {name}")
        if definition:
            texts_to_encode.append(definition)

        if not texts_to_encode:
            continue

        embeddings = model.encode(texts_to_encode)
        theme_vectors[theme_id] = np.array(embeddings)

    return theme_vectors


def classify_statement(
    statement: str,
    theme_vectors: Dict[str, np.ndarray],
    theme_id_to_info: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], float]:
    """
    Classify a single statement against pre-computed theme embeddings.

    For each theme, computes cosine similarity between the statement embedding
    and each of the theme's vectors (name + definition), takes the max per theme,
    then picks the best overall theme.

    Returns:
        (best_theme_id, best_similarity_score)
        If no themes available, returns (None, 0.0)
    """
    if not theme_vectors:
        return None, 0.0

    model = _get_model()
    stmt_emb = model.encode(statement).reshape(1, -1)

    best_theme_id: Optional[str] = None
    best_score: float = -1.0

    for theme_id, vectors in theme_vectors.items():
        sims = cosine_similarity(stmt_emb, vectors)[0]
        max_sim = float(np.max(sims))

        if max_sim > best_score:
            best_score = max_sim
            best_theme_id = theme_id

    return best_theme_id, best_score
