"""Autocomplete / suggestion endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from services.suggest import get_suggestions

logger = logging.getLogger(__name__)

router = APIRouter(tags=["suggestions"])


@router.get("/suggest")
async def suggest(
    q: str = Query(..., description="Partial query prefix"),
    limit: int = Query(default=5, ge=1, le=20),
) -> list[str]:
    """Return autocomplete suggestions for the given prefix."""
    results = get_suggestions(q, limit=limit)
    logger.info("Suggest q='%s' → %d results", q, len(results))
    return results
