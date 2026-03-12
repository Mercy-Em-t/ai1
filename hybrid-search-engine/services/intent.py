"""Keyword-based intent detection: classifies query into products / sports / events."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "products": [
        "buy", "purchase", "price", "cheap", "expensive", "headphones", "laptop",
        "macbook", "ipad", "tablet", "tv", "television", "camera", "shoes",
        "sneakers", "jeans", "jacket", "hoodie", "book", "novel", "programming",
        "clothing", "electronics", "gadget", "device", "shop", "product",
    ],
    "sports": [
        "football", "nfl", "soccer", "basketball", "nba", "tennis", "wimbledon",
        "match", "game", "league", "tournament", "team", "player", "sport",
        "training", "camp", "coach", "racket", "ball", "gym", "fitness",
        "championship", "playoff", "season",
    ],
    "events": [
        "concert", "show", "performance", "tour", "festival", "fest", "coachella",
        "sxsw", "conference", "summit", "convention", "talk", "ted", "pycon",
        "workshop", "seminar", "film", "movie", "cinema", "food", "wine",
        "burning man", "music", "gig", "live",
    ],
}


def detect_intent(query: str) -> str:
    """
    Return the best-matching domain: ``"products"``, ``"sports"``, or ``"events"``.

    Falls back to ``"products"`` when the query is ambiguous.
    """
    query_lower = query.lower()
    tokens = re.findall(r"[a-z0-9]+", query_lower)
    token_set = set(tokens)
    full_text = query_lower  # for multi-word phrases

    scores: dict[str, int] = {domain: 0 for domain in _DOMAIN_KEYWORDS}

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if " " in kw:
                if kw in full_text:
                    scores[domain] += 2
            elif kw in token_set:
                scores[domain] += 1

    best_domain = max(scores, key=lambda d: scores[d])
    # If all scores are 0, fall back to products
    if scores[best_domain] == 0:
        best_domain = "products"

    logger.debug("Intent detected for '%s': %s (scores=%s)", query, best_domain, scores)
    return best_domain
