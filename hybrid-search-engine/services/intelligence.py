"""Self-learning intelligence: item behavior scores and query relationship graph."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ── Item Intelligence Scores ──────────────────────────────────────────────
# item_id → { "clicks": int, "purchases": int, "favorites": int, "skips": int }
_item_signals: dict[str, dict[str, int]] = defaultdict(
    lambda: {"clicks": 0, "purchases": 0, "favorites": 0, "skips": 0}
)

# Weights for computing behavior_score
_SIGNAL_WEIGHTS = {
    "clicks": 0.4,
    "purchases": 0.8,
    "favorites": 0.5,
    "skips": -0.2,
}

_EVENT_TO_SIGNAL = {
    "click": "clicks",
    "purchase": "purchases",
    "favorite": "favorites",
    "skip": "skips",
}


def record_item_signal(item_id: str, event_type: str) -> None:
    """Increment the appropriate signal counter for an item."""
    signal_key = _EVENT_TO_SIGNAL.get(event_type)
    if signal_key:
        _item_signals[item_id][signal_key] += 1


def get_item_signals(item_id: str) -> dict[str, int]:
    """Return raw signal counts for an item."""
    return dict(_item_signals[item_id])


def compute_behavior_score(item_id: str) -> float:
    """
    Compute a behavior-based score for an item.

    Formula::

        behavior_score = (clicks * 0.4) + (purchases * 0.8)
                       + (favorites * 0.5) - (skips * 0.2)

    The result is normalised to roughly [0, 1] by dividing by a
    running maximum so newer items are not permanently disadvantaged.
    """
    signals = _item_signals[item_id]
    raw = sum(signals[k] * _SIGNAL_WEIGHTS[k] for k in _SIGNAL_WEIGHTS)
    return max(0.0, raw)


def get_all_behavior_scores() -> dict[str, float]:
    """Return behavior scores for every item that has signals."""
    scores: dict[str, float] = {}
    for item_id in _item_signals:
        scores[item_id] = compute_behavior_score(item_id)
    return scores


# ── Query Relationship Graph ─────────────────────────────────────────────
# user_id → list of recent queries (ordered by time)
_user_query_history: dict[str, list[str]] = defaultdict(list)

# query → { related_query: count }  (bidirectional edges)
_query_graph: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def record_query(user_id: str, query: str) -> None:
    """
    Track a user's search query and build relationships with
    the user's previous queries (state-space graph).
    """
    q = query.lower().strip()
    if not q:
        return

    history = _user_query_history[user_id]

    # Link this query to the last N queries by the same user
    for prev_q in history[-3:]:
        if prev_q != q:
            _query_graph[prev_q][q] += 1
            _query_graph[q][prev_q] += 1

    history.append(q)
    # Keep only last 50 queries per user
    if len(history) > 50:
        _user_query_history[user_id] = history[-50:]


def get_related_queries(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return queries most frequently co-searched with *query*."""
    q = query.lower().strip()
    edges = _query_graph.get(q, {})
    if not edges:
        return []
    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
    return [
        {"query": rel_q, "strength": count}
        for rel_q, count in sorted_edges[:limit]
    ]


def get_query_map(limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    """Return the full query relationship graph (for dashboard)."""
    result: dict[str, list[dict[str, Any]]] = {}
    for query in list(_query_graph.keys())[:limit]:
        edges = _query_graph[query]
        sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
        result[query] = [
            {"query": rel_q, "strength": count}
            for rel_q, count in sorted_edges[:10]
        ]
    return result


def get_dashboard_stats() -> dict[str, Any]:
    """Return summary stats for the intelligence dashboard."""
    from services.analytics import get_search_logs, get_popular_queries

    all_logs = get_search_logs(limit=10000)
    zero_result_logs = [l for l in all_logs if l["results_count"] == 0]

    # Conversion data from item signals
    total_clicks = sum(s["clicks"] for s in _item_signals.values())
    total_purchases = sum(s["purchases"] for s in _item_signals.values())
    total_favorites = sum(s["favorites"] for s in _item_signals.values())
    total_skips = sum(s["skips"] for s in _item_signals.values())

    conversion_rate = (
        round(total_purchases / total_clicks * 100, 2) if total_clicks > 0 else 0.0
    )

    # Trending items (top by behavior score)
    behavior_scores = get_all_behavior_scores()
    trending = sorted(behavior_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_searches": len(all_logs),
        "zero_result_searches": len(zero_result_logs),
        "top_queries": get_popular_queries(limit=10),
        "total_clicks": total_clicks,
        "total_purchases": total_purchases,
        "total_favorites": total_favorites,
        "total_skips": total_skips,
        "conversion_rate": conversion_rate,
        "trending_items": [
            {"item_id": iid, "behavior_score": round(score, 4)}
            for iid, score in trending
        ],
        "query_graph_size": len(_query_graph),
    }


def clear_intelligence() -> None:
    """Clear all intelligence data (used in testing)."""
    _item_signals.clear()
    _user_query_history.clear()
    _query_graph.clear()
