"""Search analytics – logs every search request and tracks zero-result queries."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# In-memory search log store
_search_logs: list[dict[str, Any]] = []


def log_search(
    query: str,
    corrected_query: Optional[str],
    domain: str,
    results_count: int,
    query_time_ms: float,
    user_id: Optional[str] = None,
) -> None:
    """Record a search request with metadata."""
    entry = {
        "query": query,
        "corrected_query": corrected_query,
        "domain": domain,
        "results_count": results_count,
        "query_time_ms": query_time_ms,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _search_logs.append(entry)

    if results_count == 0:
        logger.warning("Zero-result search: query='%s'", query)
    else:
        logger.info(
            "Search logged: query='%s' results=%d time=%.2fms",
            query, results_count, query_time_ms,
        )


def get_search_logs(
    limit: int = 100,
    zero_results_only: bool = False,
) -> list[dict[str, Any]]:
    """Return recent search logs, optionally filtered to zero-result queries."""
    logs = _search_logs
    if zero_results_only:
        logs = [log for log in logs if log["results_count"] == 0]
    return logs[-limit:]


def get_popular_queries(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most frequently searched queries."""
    from collections import Counter

    query_counts: Counter[str] = Counter()
    for log in _search_logs:
        query_counts[log["query"]] += 1

    return [
        {"query": q, "count": c}
        for q, c in query_counts.most_common(limit)
    ]


def clear_logs() -> None:
    """Clear all search logs (used in testing)."""
    _search_logs.clear()
