"""Intent prediction: transition probabilities and next-interest prediction.

Uses the query relationship graph from the intelligence layer and user
behavior sequences to predict what a user might want next.

Prediction sources:
 1. Query co-occurrence transitions (from intelligence._query_graph)
 2. Category transition patterns (from user event history)
 3. Collective patterns across all users
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def get_transition_probabilities(
    query: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Return transition probabilities for a given query.

    Uses the query relationship graph to compute the probability
    of each next query based on co-occurrence counts.

    Returns list of ``{"query": str, "probability": float}`` sorted
    by probability descending.
    """
    from services.intelligence import _query_graph

    q = query.lower().strip()
    edges = _query_graph.get(q, {})
    if not edges:
        return []

    total = sum(edges.values())
    if total == 0:
        return []

    transitions = [
        {"query": rel_q, "probability": round(count / total, 4)}
        for rel_q, count in edges.items()
    ]
    transitions.sort(key=lambda x: x["probability"], reverse=True)
    return transitions[:limit]


def get_predicted_interests(
    user_id: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Predict what a user might want next based on behavior patterns.

    Combines:
     - Recent query transitions (what commonly follows their last queries)
     - Category transitions (what categories they're drifting toward)
     - Collective intelligence (what similar users explored next)

    Returns list of ``{"interest": str, "score": float, "source": str}``
    sorted by score descending.
    """
    from services.intelligence import _user_query_history, _query_graph

    predictions: dict[str, dict[str, Any]] = {}

    # ── Source 1: Query transitions from user's recent queries ─────────
    history = _user_query_history.get(user_id, [])
    if history:
        # Weight recent queries higher
        recent = history[-5:]
        for idx, q in enumerate(recent):
            recency_weight = (idx + 1) / len(recent)  # 0.2..1.0
            edges = _query_graph.get(q, {})
            total = sum(edges.values()) or 1
            for rel_q, count in edges.items():
                if rel_q not in [h.lower() for h in recent]:
                    prob = count / total
                    score = prob * recency_weight
                    key = rel_q
                    if key in predictions:
                        predictions[key]["score"] = max(
                            predictions[key]["score"], score
                        )
                    else:
                        predictions[key] = {
                            "interest": rel_q,
                            "score": score,
                            "source": "query_transition",
                        }

    # ── Source 2: Category transitions from user events ────────────────
    from services.personalization import get_all_events

    events = get_all_events().get(user_id, [])
    if events:
        from main import item_store

        # Extract category sequence from recent events
        recent_events = events[-10:]
        categories: list[str] = []
        for ev in recent_events:
            item = item_store.get(ev.get("item_id", ""))
            if item:
                cat = (item.get("category") or "").lower()
                if cat:
                    categories.append(cat)

        # Find related categories user hasn't explored heavily
        if categories:
            from services.discovery_graph import get_related_categories

            last_cat = categories[-1]
            related = get_related_categories(last_cat, limit=limit)
            user_cats = set(categories)
            for rc in related:
                rel_cat = rc["category"]
                if rel_cat not in user_cats:
                    score = rc["strength"] * 0.3  # scale down category predictions
                    if rel_cat not in predictions or predictions[rel_cat]["score"] < score:
                        predictions[rel_cat] = {
                            "interest": rel_cat,
                            "score": round(score, 4),
                            "source": "category_transition",
                        }

    # ── Source 3: Collective intelligence (popular transitions) ────────
    if not predictions:
        # Fall back to most popular query transitions across all users
        from services.intelligence import _query_graph

        all_edges: dict[str, int] = defaultdict(int)
        for q_edges in _query_graph.values():
            for target, count in q_edges.items():
                all_edges[target] += count

        if all_edges:
            max_count = max(all_edges.values()) or 1
            for target, count in sorted(
                all_edges.items(), key=lambda x: x[1], reverse=True
            )[:limit]:
                score = round(count / max_count * 0.2, 4)  # low weight for collective
                if target not in predictions:
                    predictions[target] = {
                        "interest": target,
                        "score": score,
                        "source": "collective",
                    }

    # Sort by score and return top N
    result = sorted(predictions.values(), key=lambda x: x["score"], reverse=True)
    return result[:limit]


def clear_predictions() -> None:
    """Clear prediction state (used in testing).

    Note: predictions are derived from intelligence and personalization
    data, so clearing those clears predictions too. This function exists
    for API symmetry.
    """
    pass  # Predictions are computed on-the-fly from shared state
