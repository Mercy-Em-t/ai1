"""Discovery graph: item/category relationships, exploration scoring,
interest decay, multi-interest profiles, and curiosity selection.

Implements the balanced discovery philosophy:
  guide, don't trap · recommend, don't force · explore, don't narrow too quickly
"""
from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Category relationship graph ──────────────────────────────────────────
# category → { related_category: strength }
_category_graph: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

# Static seed relationships (bootstrapped on first call)
_SEED_RELATIONSHIPS: dict[str, list[str]] = {
    "sports": ["fitness", "wellness", "outdoors"],
    "tennis": ["padel", "squash", "badminton"],
    "padel": ["tennis", "squash", "sports"],
    "fitness": ["sports", "wellness", "yoga"],
    "wellness": ["fitness", "yoga", "spa"],
    "music": ["concerts", "festivals", "entertainment"],
    "technology": ["electronics", "gadgets", "innovation"],
    "food": ["restaurants", "cooking", "health"],
    "travel": ["tourism", "adventure", "outdoors"],
    "education": ["courses", "training", "workshops"],
}

_seeded = False


def _ensure_seed() -> None:
    """Populate the category graph with seed relationships once."""
    global _seeded
    if _seeded:
        return
    for cat, related in _SEED_RELATIONSHIPS.items():
        for rel in related:
            _category_graph[cat][rel] = max(_category_graph[cat][rel], 1.0)
            _category_graph[rel][cat] = max(_category_graph[rel][cat], 1.0)
    _seeded = True


def strengthen_category_link(cat_a: str, cat_b: str, amount: float = 0.1) -> None:
    """Strengthen the relationship between two categories (learned from user behaviour)."""
    a, b = cat_a.lower().strip(), cat_b.lower().strip()
    if a and b and a != b:
        _category_graph[a][b] += amount
        _category_graph[b][a] += amount


def get_related_categories(category: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return categories most related to *category*."""
    _ensure_seed()
    cat = category.lower().strip()
    edges = _category_graph.get(cat, {})
    if not edges:
        return []
    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
    return [
        {"category": rel_cat, "strength": round(strength, 4)}
        for rel_cat, strength in sorted_edges[:limit]
    ]


# ── Interest decay ───────────────────────────────────────────────────────

_DECAY_HALF_LIFE_DAYS = 7  # interest halves every 7 days


def _time_decay(timestamp_iso: str) -> float:
    """Return a 0–1 decay factor based on age of the event.

    Uses exponential decay with a configurable half-life.
    Recent events → ~1.0, old events → approaching 0.
    """
    try:
        event_time = datetime.fromisoformat(timestamp_iso)
    except (ValueError, TypeError):
        return 0.5  # neutral if unparseable

    # Make both times offset-aware UTC
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_ago = max(0.0, (now - event_time).total_seconds() / 86400)
    return math.exp(-0.693 * days_ago / _DECAY_HALF_LIFE_DAYS)  # ln(2) ≈ 0.693


def compute_decayed_preferences(user_id: str) -> dict[str, Any]:
    """Return category/tag scores with time-decay applied to each event.

    Recent interactions count more, older ones fade away — preventing
    the system from trapping users in past interests.
    """
    from main import item_store
    from services.personalization import get_all_events

    events = get_all_events().get(user_id, [])
    if not events:
        return {"categories": {}, "tags": {}, "event_count": 0}

    EVENT_WEIGHTS = {"click": 1, "purchase": 3, "favorite": 2, "skip": -1}
    category_scores: dict[str, float] = {}
    tag_scores: dict[str, float] = {}

    for ev in events:
        base_weight = EVENT_WEIGHTS.get(ev.get("event_type", ""), 0)
        decay = _time_decay(ev.get("timestamp", ""))
        weight = base_weight * decay

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


# ── Multi-interest profile ───────────────────────────────────────────────


def build_interest_profile(user_id: str) -> dict[str, Any]:
    """Build a multi-interest profile with normalised category weights.

    Returns a profile like::

        {
            "user_id": "u1",
            "interests": {"sports": 0.6, "travel": 0.3, "wellness": 0.4},
            "top_tags": ["tennis", "padel", "yoga"],
            "event_count": 15
        }
    """
    prefs = compute_decayed_preferences(user_id)
    categories = prefs["categories"]
    tags = prefs["tags"]

    # Normalise category scores to [0, 1]
    max_cat = max(categories.values(), default=1.0) or 1.0
    normalised = {
        cat: round(score / max_cat, 4)
        for cat, score in categories.items()
    }

    # Top tags
    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    top_tags = [t for t, _ in sorted_tags[:10]]

    return {
        "user_id": user_id,
        "interests": normalised,
        "top_tags": top_tags,
        "event_count": prefs["event_count"],
    }


# ── Exploration scoring ──────────────────────────────────────────────────


def compute_exploration_score(item: dict, user_categories: dict[str, float]) -> float:
    """Score how *exploratory* an item is for a given user.

    Items in categories the user has NOT interacted with score higher,
    encouraging diversity and preventing algorithm bubbles.

    Returns 0–1 where 1 = completely new territory.
    """
    if not user_categories:
        return 0.5  # neutral for new users

    cat = (item.get("category") or "").lower()
    if not cat:
        return 0.5

    # If user has engaged with this category, exploration score is low
    familiarity = user_categories.get(cat, 0.0)
    max_fam = max(user_categories.values(), default=1.0) or 1.0
    norm_fam = familiarity / max_fam

    # Invert: high familiarity → low exploration score
    return round(1.0 - norm_fam, 4)


# ── Curiosity selection ──────────────────────────────────────────────────


def get_curiosity_items(user_id: str, limit: int = 5) -> list[dict]:
    """Select items from categories the user has NOT explored yet.

    This is the "Explore something new" / "Unexpected finds" feature.
    """
    from main import item_store
    from ranking.ranker import rank_results

    prefs = compute_decayed_preferences(user_id)
    known_categories = set(c.lower() for c in prefs["categories"])

    all_items = list(item_store.values())
    if not all_items:
        return []

    # Find items in categories user hasn't touched
    unexplored = [
        i for i in all_items
        if (i.get("category") or "").lower() not in known_categories
    ]

    if not unexplored:
        # User has explored every category; pick least-explored
        cat_scores = prefs["categories"]
        least_explored = sorted(cat_scores.items(), key=lambda x: x[1])
        if least_explored:
            target_cat = least_explored[0][0].lower()
            unexplored = [
                i for i in all_items
                if (i.get("category") or "").lower() == target_cat
            ]

    if not unexplored:
        unexplored = all_items  # fallback

    # Rank by general popularity/quality, then sample for variety
    ranked = rank_results(unexplored)
    # Take top candidates and shuffle for serendipity
    top_pool = ranked[: max(limit * 3, 15)]
    random.shuffle(top_pool)
    selected = top_pool[:limit]

    return [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in selected
    ]


# ── Balanced discovery ranking ───────────────────────────────────────────


def compute_adaptive_exploration(
    user_id: str | None = None,
    has_exact_query: bool = False,
) -> float:
    """Dynamically compute exploration weight based on user context.

    Ratios adapt to the user's situation:

    =============================  ===========
    Situation                      Exploration
    =============================  ===========
    New user (no history)          0.35
    Returning user (some history)  0.20
    Heavy user (lots of history)   0.15
    Exact query match              0.10
    =============================  ===========
    """
    if has_exact_query:
        return 0.10  # user knows what they want

    if not user_id:
        return 0.35  # anonymous → more exploration

    from services.personalization import get_all_events

    events = get_all_events().get(user_id, [])
    event_count = len(events)

    if event_count == 0:
        return 0.35  # new user → high exploration
    elif event_count < 10:
        return 0.25  # light user → moderate exploration
    else:
        return 0.15  # returning power user → lower exploration


def discovery_rank(
    items: list[dict],
    query_scores: dict[str, float] | None = None,
    user_id: str | None = None,
    exploration_weight: float = 0.20,
) -> list[dict]:
    """Rank items with an exploration factor to ensure diversity.

    Formula::

        score = relevance × 0.40 + behavior × 0.25
              + popularity × 0.15 + exploration × 0.20

    The exploration component rewards items in categories the user
    has NOT interacted with, preventing algorithm bubbles.
    """
    if not items:
        return []

    from services.intelligence import compute_behavior_score

    query_scores = query_scores or {}

    # Get user category familiarity for exploration scoring
    user_categories: dict[str, float] = {}
    if user_id:
        prefs = compute_decayed_preferences(user_id)
        user_categories = prefs.get("categories", {})

    max_relevance = max(query_scores.values(), default=1.0) or 1.0

    raw_behavior: dict[str, float] = {}
    for item in items:
        raw_behavior[item["id"]] = compute_behavior_score(item["id"])
    max_behavior = max(raw_behavior.values(), default=1.0) or 1.0

    rel_w = 0.40
    beh_w = 0.25
    pop_w = 0.15
    exp_w = exploration_weight

    scored: list[tuple[float, dict]] = []
    for item in items:
        item_id = item["id"]
        rel = query_scores.get(item_id, 0.0) / max_relevance
        beh = raw_behavior[item_id] / max_behavior
        pop = (item.get("popularity") or 0.0) / 100.0
        exp = compute_exploration_score(item, user_categories)

        composite = rel_w * rel + beh_w * beh + pop_w * pop + exp_w * exp
        item_copy = dict(item)
        item_copy["_score"] = round(composite, 6)
        item_copy["_exploration"] = exp
        scored.append((composite, item_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


# ── Cleanup ──────────────────────────────────────────────────────────────


def clear_discovery_graph() -> None:
    """Clear all discovery graph data (used in testing)."""
    global _seeded
    _category_graph.clear()
    _seeded = False
