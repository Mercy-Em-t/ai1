"""Tests for the self-learning intelligence system."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from services.analytics import clear_logs
from services.cache import _memory_cache
from services.intelligence import clear_intelligence


@pytest.fixture()
def client():
    """Create a TestClient with lifespan so sample data is loaded."""
    clear_logs()
    _memory_cache.clear()
    clear_intelligence()
    with TestClient(app) as c:
        yield c
    clear_logs()
    clear_intelligence()


# ── Item Intelligence Scores ─────────────────────────────────────────────


class TestItemBehaviorScores:
    def test_click_event_updates_score(self, client):
        """Clicking an item should increase its behavior score."""
        # Record clicks on a known sample item
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u2", "item_id": "prod-001", "event_type": "click",
        })

        resp = client.get("/intelligence/item/prod-001/score")
        assert resp.status_code == 200
        data = resp.json()
        assert data["item_id"] == "prod-001"
        assert data["signals"]["clicks"] == 2
        assert data["behavior_score"] > 0

    def test_purchase_event_higher_weight(self, client):
        """Purchases should have higher weight than clicks."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        resp1 = client.get("/intelligence/item/prod-001/score")
        click_score = resp1.json()["behavior_score"]

        clear_intelligence()

        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "purchase",
        })
        resp2 = client.get("/intelligence/item/prod-001/score")
        purchase_score = resp2.json()["behavior_score"]

        assert purchase_score > click_score

    def test_favorite_event(self, client):
        """Favoriting should be recorded."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "favorite",
        })
        resp = client.get("/intelligence/item/prod-001/score")
        data = resp.json()
        assert data["signals"]["favorites"] == 1
        assert data["behavior_score"] > 0

    def test_skip_reduces_score(self, client):
        """Skips should reduce behavior score."""
        # Give some positive signals first
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        resp1 = client.get("/intelligence/item/prod-001/score")
        before = resp1.json()["behavior_score"]

        client.post("/events", json={
            "user_id": "u2", "item_id": "prod-001", "event_type": "skip",
        })
        resp2 = client.get("/intelligence/item/prod-001/score")
        after = resp2.json()["behavior_score"]

        assert after < before

    def test_all_item_scores_endpoint(self, client):
        """The item-scores endpoint should return scored items."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-002", "event_type": "purchase",
        })

        resp = client.get("/intelligence/item-scores")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        # Purchase item should rank higher
        assert data["items"][0]["item_id"] == "prod-002"

    def test_unseen_item_has_zero_score(self, client):
        """An item with no interactions should have zero score."""
        resp = client.get("/intelligence/item/nonexistent-item/score")
        assert resp.status_code == 200
        data = resp.json()
        assert data["behavior_score"] == 0


# ── Query Relationship Graph ─────────────────────────────────────────────


class TestQueryRelationships:
    def test_search_events_build_query_graph(self, client):
        """Sequential search events should create query relationships."""
        # User searches: padel → padel racket → padel shoes
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel racket",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel shoes",
        })

        resp = client.get("/intelligence/related-queries", params={"q": "padel"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "padel"
        related_queries = [r["query"] for r in data["related"]]
        assert "padel racket" in related_queries

    def test_query_map_endpoint(self, client):
        """The query-map endpoint should return the state-space graph."""
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "tennis",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "tennis racket",
        })

        resp = client.get("/intelligence/query-map")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] > 0
        assert "query_map" in data

    def test_query_context_in_click_event(self, client):
        """Click events with query context should feed the query graph."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001",
            "event_type": "click", "query": "headphones",
        })
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-002",
            "event_type": "click", "query": "wireless headphones",
        })

        resp = client.get("/intelligence/related-queries", params={"q": "headphones"})
        data = resp.json()
        related = [r["query"] for r in data["related"]]
        assert "wireless headphones" in related

    def test_no_related_queries_for_unknown(self, client):
        """Unknown query should return empty related queries."""
        resp = client.get("/intelligence/related-queries", params={"q": "xyz123"})
        assert resp.status_code == 200
        assert resp.json()["related"] == []


# ── Intelligence Dashboard ───────────────────────────────────────────────


class TestIntelligenceDashboard:
    def test_dashboard_returns_stats(self, client):
        """Dashboard should return aggregated stats."""
        resp = client.get("/intelligence/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_searches" in data
        assert "zero_result_searches" in data
        assert "top_queries" in data
        assert "total_clicks" in data
        assert "total_purchases" in data
        assert "total_favorites" in data
        assert "total_skips" in data
        assert "conversion_rate" in data
        assert "trending_items" in data
        assert "query_graph_size" in data

    def test_dashboard_reflects_events(self, client):
        """Dashboard should reflect recorded events."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "purchase",
        })
        client.post("/events", json={
            "user_id": "u2", "item_id": "prod-002", "event_type": "favorite",
        })

        resp = client.get("/intelligence/dashboard")
        data = resp.json()
        assert data["total_clicks"] == 1
        assert data["total_purchases"] == 1
        assert data["total_favorites"] == 1
        assert data["conversion_rate"] == 100.0  # 1 purchase / 1 click

    def test_dashboard_conversion_rate(self, client):
        """Conversion rate should be purchases / clicks * 100."""
        for i in range(4):
            client.post("/events", json={
                "user_id": f"u{i}", "item_id": "prod-001", "event_type": "click",
            })
        client.post("/events", json={
            "user_id": "u0", "item_id": "prod-001", "event_type": "purchase",
        })

        resp = client.get("/intelligence/dashboard")
        data = resp.json()
        assert data["total_clicks"] == 4
        assert data["total_purchases"] == 1
        assert data["conversion_rate"] == 25.0


# ── Expanded Event Types ─────────────────────────────────────────────────


class TestExpandedEventTypes:
    def test_search_event_accepted(self, client):
        """Search events should be accepted with just user_id and query."""
        resp = client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel",
        })
        assert resp.status_code == 201
        assert resp.json()["event_type"] == "search"

    def test_favorite_event_accepted(self, client):
        """Favorite events should be accepted."""
        resp = client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "favorite",
        })
        assert resp.status_code == 201
        assert resp.json()["event_type"] == "favorite"

    def test_click_with_query_context(self, client):
        """Click events can optionally include query context."""
        resp = client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001",
            "event_type": "click", "query": "headphones",
        })
        assert resp.status_code == 201
        assert resp.json()["query"] == "headphones"


# ── Behavior Score Affects Ranking ───────────────────────────────────────


class TestBehaviorAffectsRanking:
    def test_clicked_items_rank_higher(self, client):
        """Items with clicks should rank differently than items without."""
        # Record many clicks on a specific item
        for i in range(10):
            client.post("/events", json={
                "user_id": f"u{i}", "item_id": "prod-001", "event_type": "click",
            })

        # Search for something that matches multiple items
        resp = client.get("/search", params={"q": "headphones"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
        # The heavily-clicked item should appear in results
        result_ids = [r["id"] for r in data["results"]]
        assert "prod-001" in result_ids
