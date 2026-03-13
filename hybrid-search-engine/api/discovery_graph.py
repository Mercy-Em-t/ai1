"""Discovery Graph API: balanced discovery search, curiosity mode, and user interest profiles."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from services.discovery_graph import (
    build_interest_profile,
    compute_adaptive_exploration,
    discovery_rank,
    get_curiosity_items,
    get_related_categories,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discover", tags=["discovery-graph"])


def _get_store() -> dict[str, dict]:
    from main import item_store
    return item_store


def _clean(items: list[dict]) -> list[dict]:
    """Strip internal keys (prefixed with ``_``) from response items."""
    return [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in items
    ]


@router.get("")
async def balanced_discover(
    q: str | None = Query(default=None, description="Optional search query"),
    user_id: str | None = Query(default=None, description="User ID for personalisation"),
    limit: int = Query(default=10, ge=1, le=50),
    exploration: float | None = Query(
        default=None, ge=0.0, le=1.0,
        description="Exploration weight (0 = pure precision, 1 = max diversity). "
        "If omitted, the system adapts automatically based on user context.",
    ),
) -> dict[str, Any]:
    """Balanced discovery endpoint — returns four sections:

    * **top_results** – relevance + behaviour + exploration
    * **related_categories** – categories linked to query/user interests
    * **trending** – most popular items right now
    * **explore_new** – curiosity picks from unexplored territory

    The user always sees exact matches first (precision) while discovery
    sections guide them to new areas (exploration).  The exploration weight
    controls the balance.
    """
    from ranking.ranker import rank_results
    from services.candidates import generate_candidates
    from services.intelligence import get_related_queries
    from utils.spell import correct_query
    from search.engine import search_engine

    store = _get_store()
    all_items = list(store.values())

    # ── Adaptive exploration weight ──────────────────────────────────
    # If the caller did not specify an exploration weight, compute an
    # adaptive weight based on user context.  Explicit overrides are
    # respected as-is.
    if exploration is not None:
        effective_exploration = exploration
    else:
        effective_exploration = compute_adaptive_exploration(
            user_id=user_id,
            has_exact_query=q is not None,
        )

    # ── Top results (precision) ──────────────────────────────────────
    if q:
        vocabulary = search_engine.vocabulary
        corrected_q, _ = correct_query(q, vocabulary)
        candidates, scores = generate_candidates(corrected_q, domain=None, filters={}, limit=200)
        top_ranked = discovery_rank(
            candidates,
            query_scores=scores,
            user_id=user_id,
            exploration_weight=effective_exploration,
        )
    else:
        top_ranked = discovery_rank(
            all_items,
            user_id=user_id,
            exploration_weight=effective_exploration,
        )

    top_results = _clean(top_ranked[:limit])

    # ── Related categories ───────────────────────────────────────────
    related_cats: list[dict[str, Any]] = []
    if q:
        related_cats = get_related_categories(q.lower().strip(), limit=5)
    if not related_cats and user_id:
        profile = build_interest_profile(user_id)
        if profile["interests"]:
            top_interest = max(profile["interests"], key=profile["interests"].get)
            related_cats = get_related_categories(top_interest, limit=5)

    # ── Related queries ──────────────────────────────────────────────
    related_queries: list[dict[str, Any]] = []
    if q:
        related_queries = get_related_queries(q, limit=5)

    # ── Trending (popularity-driven) ─────────────────────────────────
    trending_ranked = rank_results(all_items)[:limit]
    trending = _clean(trending_ranked)

    # ── Explore something new (curiosity) ────────────────────────────
    explore_new: list[dict] = []
    if user_id:
        explore_new = get_curiosity_items(user_id, limit=min(limit, 5))
    else:
        # For anonymous users, pick a random sample
        import random
        pool = rank_results(all_items)[:30]
        random.shuffle(pool)
        explore_new = _clean(pool[:5])

    return {
        "top_results": top_results,
        "related_categories": related_cats,
        "related_queries": related_queries,
        "trending": trending,
        "explore_new": explore_new,
        "total_top": len(top_results),
        "query": q,
        "user_id": user_id,
        "exploration_weight": effective_exploration,
    }


@router.get("/curiosity")
async def curiosity_mode(
    user_id: str | None = Query(default=None, description="User ID for personalisation"),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Curiosity mode — intentionally show items the user has NOT explored.

    Returns items from unfamiliar categories, plus trending and
    "people nearby are trying" suggestions.
    """
    from ranking.ranker import rank_results

    store = _get_store()
    all_items = list(store.values())

    # Curiosity picks
    if user_id:
        unexpected = get_curiosity_items(user_id, limit=limit)
    else:
        import random
        pool = rank_results(all_items)[:30]
        random.shuffle(pool)
        unexpected = _clean(pool[:limit])

    # Trending nearby
    trending_ranked = rank_results(all_items)[:limit]
    trending = _clean(trending_ranked)

    return {
        "unexpected_finds": unexpected,
        "trending_now": trending,
        "user_id": user_id,
        "total": len(unexpected),
    }


@router.get("/profile/{user_id}")
async def user_interest_profile(user_id: str) -> dict[str, Any]:
    """Return the user's multi-interest profile with time-decayed weights.

    Interests fade over time so the user is never trapped in past behaviour.
    """
    profile = build_interest_profile(user_id)
    related: dict[str, list[dict]] = {}
    for cat in list(profile["interests"].keys())[:5]:
        related[cat] = get_related_categories(cat, limit=3)

    return {
        **profile,
        "related_categories": related,
    }


@router.get("/related-categories/{category}")
async def related_categories_endpoint(
    category: str,
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    """Return categories related to the given category in the discovery graph."""
    related = get_related_categories(category, limit=limit)
    return {
        "category": category,
        "related": related,
    }
