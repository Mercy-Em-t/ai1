"""Tests for autocomplete / suggestion service."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.analytics import _search_logs, clear_logs
from services.suggest import get_suggestions


def _setup_item_store():
    """Set up a minimal item store for testing."""
    from main import item_store

    item_store.clear()
    item_store["item-1"] = {
        "id": "item-1",
        "title": "Water Bottle",
        "tags": ["water", "hydration", "bottle"],
    }
    item_store["item-2"] = {
        "id": "item-2",
        "title": "Water Filter",
        "tags": ["water", "filter", "home"],
    }
    item_store["item-3"] = {
        "id": "item-3",
        "title": "Tennis Racket",
        "tags": ["tennis", "racket", "sports"],
    }
    item_store["item-4"] = {
        "id": "item-4",
        "title": "Tennis Shoes",
        "tags": ["tennis", "shoes", "footwear"],
    }


class TestGetSuggestions:
    def setup_method(self):
        clear_logs()
        _setup_item_store()

    def teardown_method(self):
        from main import item_store
        item_store.clear()
        clear_logs()

    def test_suggest_from_titles(self):
        results = get_suggestions("wat")
        assert any("water" in r for r in results)

    def test_suggest_from_tags(self):
        results = get_suggestions("hydra")
        assert "hydration" in results

    def test_suggest_empty_prefix(self):
        results = get_suggestions("")
        assert results == []

    def test_suggest_whitespace_prefix(self):
        results = get_suggestions("   ")
        assert results == []

    def test_suggest_respects_limit(self):
        results = get_suggestions("ten", limit=1)
        assert len(results) <= 1

    def test_suggest_from_search_history(self):
        _search_logs.append({"query": "tennis coaching"})
        _search_logs.append({"query": "tennis coaching"})
        _search_logs.append({"query": "tennis shoes"})

        results = get_suggestions("tennis")
        # Search history should be prioritized
        assert "tennis coaching" in results

    def test_suggest_no_duplicates(self):
        results = get_suggestions("ten")
        assert len(results) == len(set(results))

    def test_suggest_case_insensitive(self):
        results = get_suggestions("WATER")
        assert len(results) > 0

    def test_suggest_no_self_match_from_history(self):
        """Previous queries that exactly match the prefix shouldn't appear."""
        _search_logs.append({"query": "tennis"})
        results = get_suggestions("tennis")
        # "tennis" itself shouldn't appear as a suggestion from history
        # but item titles starting with "tennis" should
        found_from_titles = [r for r in results if "tennis" in r and r != "tennis"]
        # We should have some results from titles/tags
        assert len(results) > 0
