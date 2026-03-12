"""Full hybrid search pipeline endpoint."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, Query

from filters.filter import apply_filters
from ranking.ranker import personalized_ranking, rank_results
from services.analytics import log_search
from services.cache import get_cache, set_cache
from services.candidates import generate_candidates
from services.intent import detect_intent
from services.personalization import get_all_events
from utils.spell import correct_query
from search.engine import search_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _build_cache_key(params: dict) -> str:
    serialized = json.dumps(params, sort_keys=True, default=str)
    return "search:" + hashlib.md5(serialized.encode()).hexdigest()


@router.get("/search")
async def search(
    q: str = Query(..., description="Search query"),
    location: str | None = Query(default=None),
    min_price: float | None = Query(default=None),
    max_price: float | None = Query(default=None),
    category: str | None = Query(default=None),
    availability: bool | None = Query(default=None),
    min_date: str | None = Query(default=None),
    max_date: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    user_id: str | None = Query(default=None),
) -> dict[str, Any]:
    start_time = time.perf_counter()

    cache_key = _build_cache_key(
        {
            "q": q,
            "location": location,
            "min_price": min_price,
            "max_price": max_price,
            "category": category,
            "availability": availability,
            "min_date": min_date,
            "max_date": max_date,
            "page": page,
            "page_size": page_size,
            "user_id": user_id,
        }
    )

    cached = get_cache(cache_key)
    if cached is not None:
        logger.info("Cache hit for query '%s'", q)
        cached["query_time_ms"] = round(
            (time.perf_counter() - start_time) * 1000, 2
        )
        return cached

    logger.info("=== Search pipeline start | query='%s' ===", q)

    # ── Phase 1: Spell correction ──────────────────────────────────────────
    vocabulary = search_engine.vocabulary
    corrected_q, was_corrected = correct_query(q, vocabulary)
    corrected_display = corrected_q if was_corrected else None
    effective_query = corrected_q
    logger.info(
        "Phase 1 – Spell: corrected=%s query='%s'", was_corrected, effective_query
    )

    # ── Phase 2: Intent detection ──────────────────────────────────────────
    domain = detect_intent(effective_query)
    logger.info("Phase 2 – Intent: domain='%s'", domain)

    # ── Phase 3 & 4: Synonym expansion + candidate generation ─────────────
    candidates, scores = generate_candidates(
        effective_query, domain, filters={}, limit=200
    )
    logger.info("Phase 3/4 – Candidates: %d items", len(candidates))

    # ── Phase 5: Filtering ─────────────────────────────────────────────────
    filters: dict = {}
    if location is not None:
        filters["location"] = location
    if min_price is not None:
        filters["min_price"] = min_price
    if max_price is not None:
        filters["max_price"] = max_price
    if category is not None:
        filters["category"] = category
    if availability is not None:
        filters["availability"] = availability
    if min_date is not None:
        filters["min_date"] = min_date
    if max_date is not None:
        filters["max_date"] = max_date

    filtered = apply_filters(candidates, filters)
    logger.info("Phase 5 – Filtered: %d items remain", len(filtered))

    # ── Phase 6: Ranking ──────────────────────────────────────────────────
    ranked = rank_results(filtered, query_scores=scores)
    logger.info("Phase 6 – Ranked: %d items", len(ranked))

    # ── Phase 7: Personalisation ──────────────────────────────────────────
    if user_id:
        user_events = get_all_events()
        ranked = personalized_ranking(user_id, ranked, user_events)
        logger.info("Phase 7 – Personalised for user '%s'", user_id)

    # ── Phase 8: Pagination ───────────────────────────────────────────────
    total = len(ranked)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_items = ranked[start_idx:end_idx]

    # ── Phase 9: Response formatting ──────────────────────────────────────
    results = []
    for item in page_items:
        item_out = {k: v for k, v in item.items() if not k.startswith("_")}
        item_out["score"] = item.get("_score")
        results.append(item_out)

    query_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(
        "=== Search pipeline end | results=%d total=%d time=%.2fms ===",
        len(results), total, query_time_ms,
    )

    response: dict[str, Any] = {
        "results": results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "corrected_query": corrected_display,
        "domain": domain,
        "query_time_ms": query_time_ms,
    }

    # Cache everything except query_time_ms
    cacheable = {k: v for k, v in response.items() if k != "query_time_ms"}
    set_cache(cache_key, cacheable, ttl=300)

    # ── Analytics logging ─────────────────────────────────────────────────
    log_search(
        query=q,
        corrected_query=corrected_display,
        domain=domain,
        results_count=total,
        query_time_ms=query_time_ms,
        user_id=user_id,
    )

    return response
