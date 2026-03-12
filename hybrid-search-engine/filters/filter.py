"""Filtering module: narrows candidate list by location, dates, price, availability, category."""
from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return None


def apply_filters(candidates: list[dict], filters: dict) -> list[dict]:
    """
    Apply structured filters to a candidate list.

    Supported filter keys:
    - ``location``     : str  – substring match (case-insensitive)
    - ``min_date``     : str  – ISO date, inclusive lower bound
    - ``max_date``     : str  – ISO date, inclusive upper bound
    - ``min_price``    : float
    - ``max_price``    : float
    - ``availability`` : bool
    - ``category``     : str  – exact match (case-insensitive)
    """
    if not filters:
        return candidates

    location_filter: str | None = filters.get("location")
    min_date: date | None = _parse_date(filters.get("min_date"))
    max_date: date | None = _parse_date(filters.get("max_date"))
    min_price: float | None = filters.get("min_price")
    max_price: float | None = filters.get("max_price")
    availability_filter: bool | None = filters.get("availability")
    category_filter: str | None = filters.get("category")

    filtered: list[dict] = []

    for item in candidates:
        # Location filter
        if location_filter:
            item_location = item.get("location") or ""
            if location_filter.lower() not in item_location.lower():
                continue

        # Date filters
        if min_date or max_date:
            item_date = _parse_date(item.get("date"))
            if item_date is None:
                # Items without a date are excluded when date filtering is active
                continue
            if min_date and item_date < min_date:
                continue
            if max_date and item_date > max_date:
                continue

        # Price filters
        item_price = item.get("price")
        if min_price is not None:
            if item_price is None or item_price < min_price:
                continue
        if max_price is not None:
            if item_price is None or item_price > max_price:
                continue

        # Availability filter
        if availability_filter is not None:
            if bool(item.get("availability")) != availability_filter:
                continue

        # Category filter
        if category_filter:
            if (item.get("category") or "").lower() != category_filter.lower():
                continue

        filtered.append(item)

    logger.debug(
        "Filters reduced candidates from %d to %d", len(candidates), len(filtered)
    )
    return filtered
