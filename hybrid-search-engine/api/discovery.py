"""Discovery endpoints: trending, explore, and recommendations."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from models.item import ItemResponse
from ranking.ranker import rank_results
from services.personalization import get_all_events

logger = logging.getLogger(__name__)

router = APIRouter(tags=["discovery"])


def _get_store() -> dict[str, dict]:
    from main import item_store
    return item_store


@router.get("/trending")
async def trending(
    location: str | None = Query(default=None, description="Filter by location (substring match)"),
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Return trending items: most popular products, upcoming events, and
    trending activities.  Optionally filtered by location.
    """
    store = _get_store()
    items = list(store.values())

    # Optional location filter (case-insensitive substring)
    if location:
        loc_lower = location.lower()
        items = [
            i for i in items
            if (i.get("location") or "").lower().find(loc_lower) >= 0
        ]

    # Rank by popularity (use rank_results with no query scores so
    # popularity/rating/recency dominate)
    ranked = rank_results(items)

    # Group results by type
    products = [i for i in ranked if i.get("type") == "product"][:limit]
    events = [i for i in ranked if i.get("type") == "event"][:limit]
    services = [i for i in ranked if i.get("type") == "service"][:limit]
    venues = [i for i in ranked if i.get("type") == "venue"][:limit]
    courses = [i for i in ranked if i.get("type") == "course"][:limit]
    experiences = [i for i in ranked if i.get("type") == "experience"][:limit]

    def _clean(items: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in item.items() if not k.startswith("_")}
            for item in items
        ]

    return {
        "products": _clean(products),
        "events": _clean(events),
        "services": _clean(services),
        "venues": _clean(venues),
        "courses": _clean(courses),
        "experiences": _clean(experiences),
        "total": len(items),
        "location_filter": location,
    }


@router.get("/explore/{category}")
async def explore_category(
    category: str,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """
    Explore a category – returns all listing types within that category.

    Example: ``/explore/sports`` returns tournaments, equipment, coaching,
    venues all under the sports umbrella.
    """
    store = _get_store()
    category_lower = category.lower()

    # Match items by category OR by tags containing the category term
    matched = [
        i for i in store.values()
        if (i.get("category") or "").lower() == category_lower
        or category_lower in [t.lower() for t in i.get("tags", [])]
    ]

    ranked = rank_results(matched)

    # Group by type
    grouped: dict[str, list[dict]] = {}
    for item in ranked:
        item_type = item.get("type", "other")
        if item_type not in grouped:
            grouped[item_type] = []
        if len(grouped[item_type]) < limit:
            clean = {k: v for k, v in item.items() if not k.startswith("_")}
            grouped[item_type].append(clean)

    return {
        "category": category,
        "groups": grouped,
        "total": len(matched),
    }


@router.get("/recommendations")
async def recommendations(
    user_id: str = Query(..., description="User ID for personalised recommendations"),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """
    Return personalised recommendations based on user's search history,
    clicks, and purchases.
    """
    from ranking.ranker import personalized_ranking

    store = _get_store()
    all_items = list(store.values())
    user_events = get_all_events()

    # First rank all items generically
    ranked = rank_results(all_items)

    # Then apply personalisation boost
    personalised = personalized_ranking(user_id, ranked, user_events)

    # Take top N
    top_items = personalised[:limit]
    results = [
        {k: v for k, v in item.items() if not k.startswith("_")}
        for item in top_items
    ]

    return {
        "user_id": user_id,
        "recommendations": results,
        "total": len(results),
    }
