"""Search analytics endpoints – view search logs and stats."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from services.analytics import get_popular_queries, get_search_logs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/searches")
async def list_search_logs(
    limit: int = Query(default=100, ge=1, le=1000),
    zero_results_only: bool = Query(default=False),
) -> list[dict[str, Any]]:
    """Return recent search logs, optionally filtered to zero-result queries."""
    logs = get_search_logs(limit=limit, zero_results_only=zero_results_only)
    return logs


@router.get("/popular")
async def popular_queries(
    limit: int = Query(default=10, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Return the most frequently searched queries."""
    return get_popular_queries(limit=limit)
