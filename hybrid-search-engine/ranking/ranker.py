"""Scoring and ranking logic combining text relevance, popularity, rating, and recency."""
from __future__ import annotations

import logging
import math
from datetime import date

logger = logging.getLogger(__name__)

_TODAY = date.today()
_MAX_DATE_DELTA_DAYS = 365 * 2  # normalise recency over a 2-year window


def _recency_score(date_str: str | None) -> float:
    """Return a 0–1 recency score. Recent/upcoming dates score higher."""
    if not date_str:
        return 0.5  # neutral for items without dates
    try:
        item_date = date.fromisoformat(date_str)
    except ValueError:
        return 0.5
    delta = abs((item_date - _TODAY).days)
    score = max(0.0, 1.0 - delta / _MAX_DATE_DELTA_DAYS)
    return score


def rank_results(
    items: list[dict],
    query_scores: dict[str, float] | None = None,
) -> list[dict]:
    """
    Rank items by a weighted combination of:
    - text relevance    (35 %)
    - popularity        (25 %)
    - rating            (20 %)
    - behavior_score    (10 %)
    - recency           (10 %)

    Returns items sorted descending by composite score; each item dict
    gets an injected ``_score`` key.
    """
    if not items:
        return []

    from services.intelligence import compute_behavior_score

    query_scores = query_scores or {}

    # Normalise relevance scores to [0, 1]
    max_relevance = max(query_scores.values(), default=1.0) or 1.0

    # Collect raw behavior scores and find max for normalisation
    raw_behavior: dict[str, float] = {}
    for item in items:
        raw_behavior[item["id"]] = compute_behavior_score(item["id"])
    max_behavior = max(raw_behavior.values(), default=1.0) or 1.0

    scored: list[tuple[float, dict]] = []
    for item in items:
        item_id = item["id"]
        rel = query_scores.get(item_id, 0.0) / max_relevance
        pop = (item.get("popularity") or 0.0) / 100.0
        rat = (item.get("rating") or 0.0) / 5.0
        beh = raw_behavior[item_id] / max_behavior
        rec = _recency_score(item.get("date"))

        composite = 0.35 * rel + 0.25 * pop + 0.20 * rat + 0.10 * beh + 0.10 * rec
        item_copy = dict(item)
        item_copy["_score"] = round(composite, 6)
        scored.append((composite, item_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    ranked = [item for _, item in scored]
    logger.debug("Ranked %d items", len(ranked))
    return ranked


def personalized_ranking(
    user_id: str,
    items: list[dict],
    user_events: dict,
) -> list[dict]:
    """
    Re-rank *items* by boosting categories and tags the user has interacted
    with (click → +0.05, purchase → +0.12, skip → −0.05).

    *user_events* format::

        {
            "user123": [
                {"item_id": "prod-001", "event_type": "click"},
                ...
            ]
        }
    """
    if not user_id or not items:
        return items

    events = user_events.get(user_id, [])
    if not events:
        return items

    from main import item_store  # deferred to avoid circular import

    category_boost: dict[str, float] = {}
    tag_boost: dict[str, float] = {}

    EVENT_WEIGHTS = {"click": 0.05, "purchase": 0.12, "favorite": 0.08, "skip": -0.05}

    for ev in events:
        weight = EVENT_WEIGHTS.get(ev.get("event_type", ""), 0.0)
        interacted = item_store.get(ev.get("item_id", ""))
        if interacted is None:
            continue
        cat = interacted.get("category", "")
        if cat:
            category_boost[cat] = category_boost.get(cat, 0.0) + weight
        for tag in interacted.get("tags", []):
            tag_boost[tag] = tag_boost.get(tag, 0.0) + weight

    re_scored: list[tuple[float, dict]] = []
    for item in items:
        base = item.get("_score", 0.0)
        boost = category_boost.get(item.get("category", ""), 0.0)
        for tag in item.get("tags", []):
            boost += tag_boost.get(tag, 0.0) * 0.5  # tag boost is halved
        new_score = max(0.0, base + boost)
        item_copy = dict(item)
        item_copy["_score"] = round(new_score, 6)
        re_scored.append((new_score, item_copy))

    re_scored.sort(key=lambda x: x[0], reverse=True)
    result = [item for _, item in re_scored]
    logger.debug(
        "Personalised ranking for user '%s': %d items re-ranked", user_id, len(result)
    )
    return result
