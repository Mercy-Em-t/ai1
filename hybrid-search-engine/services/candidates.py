"""Candidate generation: synonym expansion + search engine lookup + semantic search."""
from __future__ import annotations

import logging

from search.engine import search_engine
from utils.synonyms import expand_query

logger = logging.getLogger(__name__)


def _semantic_candidates(
    query: str,
    limit: int,
) -> dict[str, float]:
    """Generate candidates using vector similarity (if embeddings are available)."""
    from main import item_store
    from services.embeddings import cosine_similarity, encode_text

    query_vector = encode_text(query)
    if query_vector is None:
        return {}

    scores: dict[str, float] = {}
    for item_id, item in item_store.items():
        item_vector = item.get("vector")
        if item_vector is None:
            continue
        sim = cosine_similarity(query_vector, item_vector)
        if sim > 0.0:
            scores[item_id] = sim

    # Return top-N by similarity
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked[:limit])


def generate_candidates(
    query: str,
    domain: str,
    filters: dict,
    limit: int = 200,
) -> tuple[list[dict], dict[str, float]]:
    """
    Generate candidate items for the search pipeline.

    1. Expand the query with synonyms.
    2. Run each variation through the search engine.
    3. Optionally add semantic (vector) candidates.
    4. Merge scores (taking the maximum for each item).
    5. Return ``(candidates, scores)`` where *candidates* is a list of item
       dicts and *scores* maps item_id → relevance score.
    """
    from main import item_store  # deferred import to avoid circular refs

    variations = expand_query(query, domain)
    logger.info(
        "Generating candidates for query='%s' domain='%s' variations=%d",
        query, domain, len(variations),
    )

    merged_scores: dict[str, float] = {}

    for variation in variations:
        results = search_engine.search(variation, domain=domain, limit=limit)
        for item_id, score in results:
            if item_id not in merged_scores or score > merged_scores[item_id]:
                merged_scores[item_id] = score

    # ── Semantic search candidates ────────────────────────────────────────
    semantic_scores = _semantic_candidates(query, limit=limit)
    if semantic_scores:
        logger.info("Semantic search returned %d candidates", len(semantic_scores))
        for item_id, sim_score in semantic_scores.items():
            # Blend: keep the higher of TF-IDF or semantic score
            if item_id not in merged_scores or sim_score > merged_scores[item_id]:
                merged_scores[item_id] = sim_score

    # Retrieve item dicts from the store
    candidates: list[dict] = []
    for item_id in merged_scores:
        item = item_store.get(item_id)
        if item is not None:
            candidates.append(item)

    logger.info("Generated %d candidates for query='%s'", len(candidates), query)
    return candidates, merged_scores
