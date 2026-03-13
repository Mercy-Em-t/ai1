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
from services.intelligence import record_unmet_demand
from services.personalization import get_all_events
from utils.spell import correct_query
from search.engine import search_engine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


def _build_cache_key(params: dict) -> str:
    serialized = json.dumps(params, sort_keys=True, default=str)
    return "search:" + hashlib.md5(serialized.encode()).hexdigest()


def _zero_result_recovery(
    query: str,
    domain: str | None,
    page_size: int,
) -> tuple[list[dict], list[str]]:
    """
    Graceful recovery when the primary search pipeline returns zero results.

    Returns ``(fallback_results, suggestions)`` where:
    - *fallback_results* are the closest items found via knowledge graph
      expansion and category broadening.
    - *suggestions* are alternative query strings the user might try.
    """
    from main import item_store
    from services.knowledge_graph import expand_query_via_kg, get_neighbors

    fallback_ids: dict[str, float] = {}
    suggestions: list[str] = []

    # ── Strategy 1: Knowledge graph expansion ──────────────────────────
    try:
        kg_items = expand_query_via_kg(query, limit=page_size * 2)
        for exp in kg_items:
            iid = exp["item_id"]
            fallback_ids[iid] = max(fallback_ids.get(iid, 0.0), exp["kg_weight"])
    except Exception:
        pass

    # ── Strategy 2: Category broadening via KG neighbours ──────────────
    try:
        q_lower = query.lower().strip()
        cat_node_id = f"cat:{q_lower}"
        neighbors = get_neighbors(cat_node_id, limit=10)
        for nb in neighbors:
            if nb["node_type"] == "category":
                cat_label = nb["label"]
                if cat_label.lower() != q_lower:
                    suggestions.append(cat_label)
                # Follow to items in related categories
                cat_items = get_neighbors(nb["node_id"], limit=page_size)
                for ci in cat_items:
                    if ci["node_type"] == "item":
                        raw_id = ci["node_id"].replace("item:", "")
                        fallback_ids[raw_id] = max(
                            fallback_ids.get(raw_id, 0.0), 0.1
                        )
    except Exception:
        pass

    # ── Strategy 3: Semantic search fallback ──────────────────────────
    try:
        from services.candidates import _semantic_candidates

        sem = _semantic_candidates(query, limit=page_size)
        for iid, sim in sem.items():
            if sim > 0.25:
                fallback_ids[iid] = max(fallback_ids.get(iid, 0.0), sim)
    except Exception:
        pass

    # ── Build suggestion strings from fallback item titles ────────────
    seen_titles: set[str] = set()
    for iid in sorted(fallback_ids, key=fallback_ids.get, reverse=True):
        item = item_store.get(iid)
        if item:
            title = item.get("title", "")
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                suggestions.append(title)
        if len(suggestions) >= 5:
            break

    # Deduplicate suggestions while preserving order
    unique_suggestions: list[str] = []
    seen: set[str] = set()
    for s in suggestions:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique_suggestions.append(s)

    # Resolve fallback items
    sorted_fallback = sorted(fallback_ids.items(), key=lambda x: x[1], reverse=True)
    fallback_results: list[dict] = []
    for iid, score in sorted_fallback[:page_size]:
        item = item_store.get(iid)
        if item:
            item_out = {k: v for k, v in item.items() if not k.startswith("_")}
            item_out["score"] = round(score, 4)
            fallback_results.append(item_out)

    return fallback_results, unique_suggestions[:5]


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

    # ── Phase 10: Graceful zero-result recovery ───────────────────────────
    fallback_results: list[dict] = []
    suggestions: list[str] = []

    if total == 0:
        logger.info("Phase 10 – Zero results: triggering graceful recovery")
        fallback_results, suggestions = _zero_result_recovery(
            effective_query, domain, page_size
        )
        record_unmet_demand(q)
        logger.info(
            "Phase 10 – Recovery: %d fallback results, %d suggestions",
            len(fallback_results), len(suggestions),
        )

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
        "suggestions": suggestions,
        "fallback_results": fallback_results,
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
