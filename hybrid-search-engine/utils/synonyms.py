"""Domain-specific synonym expansion."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYNONYM_MAP: dict[str, list[str]] = {
    # Products – electronics
    "headphones": ["earphones", "earbuds", "audio"],
    "laptop": ["notebook", "computer", "macbook"],
    "tv": ["television", "screen", "display"],
    "camera": ["photography", "photo", "dslr"],
    "tablet": ["ipad", "touchscreen"],
    # Products – clothing
    "shoes": ["sneakers", "footwear", "boots"],
    "jeans": ["pants", "denim", "trousers"],
    "jacket": ["coat", "outerwear", "hoodie"],
    # Products – books
    "programming": ["coding", "software", "developer"],
    "fiction": ["novel", "story", "literature"],
    "self-help": ["productivity", "habits", "motivation"],
    # Sports – football
    "football": ["nfl", "soccer", "american-football"],
    "nfl": ["football", "american-football", "gridiron"],
    "super bowl": ["championship", "finals", "nfl"],
    # Sports – basketball
    "basketball": ["nba", "hoops", "bball"],
    "nba": ["basketball", "hoops"],
    "dunk": ["slam-dunk", "basketball"],
    # Sports – tennis
    "tennis": ["racket", "wimbledon", "grand-slam"],
    "wimbledon": ["tennis", "grass-court", "grand-slam"],
    # Events – concerts
    "concert": ["show", "performance", "live", "gig"],
    "music": ["concert", "performance", "festival"],
    "festival": ["fest", "event", "outdoor"],
    "tour": ["concert", "show", "performance"],
    # Events – conferences
    "conference": ["summit", "symposium", "convention", "event"],
    "tech": ["technology", "software", "developer"],
    "summit": ["conference", "event", "gathering"],
    # Events – festivals
    "film": ["movie", "cinema", "screening"],
    "food": ["culinary", "dining", "gastronomy"],
    "art": ["arts", "creative", "culture"],
}

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "products": ["electronics", "clothing", "books", "buy", "purchase", "shop", "price"],
    "sports": ["football", "basketball", "tennis", "game", "match", "play", "team", "league"],
    "events": ["concert", "festival", "conference", "event", "show", "music", "film"],
}


def expand_query(query: str, domain: str | None = None) -> list[str]:
    """
    Return a list of query variations by expanding synonyms.

    The original query is always the first element.
    *domain* is used to prioritise domain-relevant synonyms.
    """
    variations: list[str] = [query]
    query_lower = query.lower()

    for term, synonyms in _SYNONYM_MAP.items():
        if term in query_lower:
            for synonym in synonyms:
                new_query = query_lower.replace(term, synonym)
                if new_query not in variations:
                    variations.append(new_query)

    # Add individual synonym tokens that aren't already present
    extra_terms: list[str] = []
    for term, synonyms in _SYNONYM_MAP.items():
        if term in query_lower:
            extra_terms.extend(s for s in synonyms if s not in query_lower)

    if extra_terms:
        expanded = query + " " + " ".join(extra_terms)
        if expanded not in variations:
            variations.append(expanded)

    logger.debug(
        "Query '%s' expanded to %d variations (domain=%s)",
        query, len(variations), domain,
    )
    return variations
