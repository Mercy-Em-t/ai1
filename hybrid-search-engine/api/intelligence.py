"""Intelligence dashboard and query insights endpoints."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from services.intelligence import (
    compute_behavior_score,
    get_all_behavior_scores,
    get_dashboard_stats,
    get_item_signals,
    get_query_map,
    get_related_queries,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


@router.get("/dashboard")
async def dashboard() -> dict[str, Any]:
    """
    Intelligence dashboard — aggregated stats about user behavior,
    trending items, conversion rates, and query relationships.
    """
    return get_dashboard_stats()


@router.get("/item/{item_id}/score")
async def item_score(item_id: str) -> dict[str, Any]:
    """Return behavior signals and computed score for a single item."""
    signals = get_item_signals(item_id)
    score = compute_behavior_score(item_id)
    return {
        "item_id": item_id,
        "signals": signals,
        "behavior_score": round(score, 4),
    }


@router.get("/item-scores")
async def all_item_scores(
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Return behavior scores for all items, sorted by score descending."""
    scores = get_all_behavior_scores()
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "items": [
            {"item_id": iid, "behavior_score": round(s, 4)}
            for iid, s in sorted_scores
        ],
        "total": len(scores),
    }


@router.get("/related-queries")
async def related_queries_endpoint(
    q: str = Query(..., description="Query to find related searches for"),
    limit: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    """Return queries most frequently co-searched with the given query."""
    related = get_related_queries(q, limit=limit)
    return {
        "query": q,
        "related": related,
    }


@router.get("/query-map")
async def query_map(
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    """Return the query relationship graph (state-space map)."""
    graph = get_query_map(limit=limit)
    return {
        "query_map": graph,
        "total_nodes": len(graph),
    }
