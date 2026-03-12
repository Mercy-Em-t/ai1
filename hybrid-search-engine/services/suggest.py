"""Autocomplete / suggestion service – returns completions for partial queries."""
from __future__ import annotations

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def get_suggestions(
    prefix: str,
    limit: int = 5,
) -> list[str]:
    """
    Return autocomplete suggestions for the given prefix.

    Sources (in priority order):
    1. Popular previous search queries
    2. Item titles
    3. Item tags
    """
    from main import item_store
    from services.analytics import _search_logs

    prefix_lower = prefix.strip().lower()
    if not prefix_lower:
        return []

    seen: set[str] = set()
    results: list[str] = []

    # ── Source 1: previous searches (ranked by frequency) ─────────────────
    query_counts: Counter[str] = Counter()
    for log in _search_logs:
        q = log["query"].lower()
        if q.startswith(prefix_lower) and q != prefix_lower:
            query_counts[q] += 1

    for q, _ in query_counts.most_common(limit):
        if q not in seen:
            seen.add(q)
            results.append(q)

    # ── Source 2: item titles ─────────────────────────────────────────────
    for item in item_store.values():
        title = item.get("title", "").lower()
        if title.startswith(prefix_lower) and title not in seen:
            seen.add(title)
            results.append(title)
        # Also match individual words in the title
        for word in title.split():
            if word.startswith(prefix_lower) and word not in seen:
                seen.add(word)
                results.append(word)

    # ── Source 3: item tags ───────────────────────────────────────────────
    for item in item_store.values():
        for tag in item.get("tags", []):
            tag_lower = tag.lower()
            if tag_lower.startswith(prefix_lower) and tag_lower not in seen:
                seen.add(tag_lower)
                results.append(tag_lower)

    return results[:limit]
