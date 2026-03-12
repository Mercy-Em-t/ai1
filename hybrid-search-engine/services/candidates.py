"""Candidate generation: synonym expansion + search engine lookup."""
from __future__ import annotations

import logging

from search.engine import search_engine
from utils.synonyms import expand_query

logger = logging.getLogger(__name__)


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
    3. Merge scores (taking the maximum for each item).
    4. Return ``(candidates, scores)`` where *candidates* is a list of item
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

    # Retrieve item dicts from the store
    candidates: list[dict] = []
    for item_id in merged_scores:
        item = item_store.get(item_id)
        if item is not None:
            candidates.append(item)

    logger.info("Generated %d candidates for query='%s'", len(candidates), query)
    return candidates, merged_scores
