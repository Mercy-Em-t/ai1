"""User event history and preference management."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# user_id → list of event dicts
_user_events: dict[str, list[dict]] = {}


def record_event(user_id: str, item_id: str, event_type: str, query: str | None = None) -> None:
    """Persist a user interaction in the in-memory store."""
    if user_id not in _user_events:
        _user_events[user_id] = []
    event: dict = {
        "item_id": item_id,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if query:
        event["query"] = query
    _user_events[user_id].append(event)

    # Feed intelligence layer
    from services.intelligence import record_item_signal, record_query
    if item_id and event_type in ("click", "purchase", "favorite", "skip"):
        record_item_signal(item_id, event_type)
    if query:
        record_query(user_id, query)

    # Feed discovery graph — strengthen category links from user paths
    from services.discovery_graph import strengthen_category_link
    if item_id:
        from main import item_store
        current_item = item_store.get(item_id)
        if current_item:
            current_cat = (current_item.get("category") or "").lower()
            # Link current item's category to categories of recent interactions
            recent = _user_events.get(user_id, [])[-5:]
            for prev_ev in recent[:-1]:
                prev_item = item_store.get(prev_ev.get("item_id", ""))
                if prev_item:
                    prev_cat = (prev_item.get("category") or "").lower()
                    if prev_cat and current_cat and prev_cat != current_cat:
                        strengthen_category_link(prev_cat, current_cat, 0.1)

    logger.info(
        "Recorded event: user=%s item=%s type=%s query=%s",
        user_id, item_id, event_type, query,
    )


def get_user_preferences(user_id: str) -> dict:
    """
    Return a summary of what categories and tags the user likes/dislikes
    based on their event history.
    """
    from main import item_store  # deferred import

    events = _user_events.get(user_id, [])
    if not events:
        return {"categories": {}, "tags": {}, "event_count": 0}

    EVENT_WEIGHTS = {"click": 1, "purchase": 3, "favorite": 2, "skip": -1}
    category_scores: dict[str, float] = {}
    tag_scores: dict[str, float] = {}

    for ev in events:
        weight = EVENT_WEIGHTS.get(ev.get("event_type", ""), 0)
        item = item_store.get(ev.get("item_id", ""))
        if item is None:
            continue
        cat = item.get("category", "")
        if cat:
            category_scores[cat] = category_scores.get(cat, 0.0) + weight
        for tag in item.get("tags", []):
            tag_scores[tag] = tag_scores.get(tag, 0.0) + weight

    return {
        "categories": category_scores,
        "tags": tag_scores,
        "event_count": len(events),
    }


def get_all_events() -> dict[str, list[dict]]:
    """Return the raw event store (used by the ranker)."""
    return _user_events
