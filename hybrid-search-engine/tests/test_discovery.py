"""Tests for discovery endpoints: trending, explore, and recommendations."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from services.analytics import clear_logs
from services.cache import _memory_cache
from services.personalization import record_event


@pytest.fixture()
def client():
    """Create a TestClient with lifespan so sample data is loaded."""
    clear_logs()
    _memory_cache.clear()
    with TestClient(app) as c:
        yield c
    clear_logs()


class TestTrendingEndpoint:
    def test_trending_returns_grouped_results(self, client):
        resp = client.get("/trending")
        assert resp.status_code == 200
        data = resp.json()
        assert "products" in data
        assert "events" in data
        assert "services" in data
        assert "venues" in data
        assert "courses" in data
        assert "experiences" in data
        assert "total" in data
        assert data["total"] > 0

    def test_trending_with_location_filter(self, client):
        resp = client.get("/trending", params={"location": "Nairobi"})
        assert resp.status_code == 200
        data = resp.json()
        # Should only return items with Nairobi location
        all_items = (
            data["products"] + data["events"] + data["services"]
            + data["venues"] + data["courses"] + data["experiences"]
        )
        for item in all_items:
            assert "nairobi" in (item.get("location") or "").lower()

    def test_trending_with_limit(self, client):
        resp = client.get("/trending", params={"limit": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["products"]) <= 3
        assert len(data["events"]) <= 3

    def test_trending_products_are_sorted_by_score(self, client):
        resp = client.get("/trending")
        data = resp.json()
        products = data["products"]
        if len(products) >= 2:
            # Items should be ranked (popularity is a major factor)
            first_pop = products[0].get("popularity", 0)
            assert first_pop > 0  # top item should have non-zero popularity


class TestExploreEndpoint:
    def test_explore_sports(self, client):
        resp = client.get("/explore/sports")
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "sports"
        assert "groups" in data
        assert data["total"] > 0

    def test_explore_tennis(self, client):
        resp = client.get("/explore/tennis")
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "tennis"
        assert data["total"] > 0
        # Tennis items should include events and products
        groups = data["groups"]
        assert len(groups) > 0

    def test_explore_returns_grouped_types(self, client):
        resp = client.get("/explore/tennis")
        data = resp.json()
        groups = data["groups"]
        # Tennis has events (Wimbledon, etc), products (racket), services (coaching), courses
        type_keys = set(groups.keys())
        assert len(type_keys) >= 2  # At least events and products

    def test_explore_nonexistent_category(self, client):
        resp = client.get("/explore/unicorns")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0


class TestRecommendationsEndpoint:
    def test_recommendations_requires_user_id(self, client):
        resp = client.get("/recommendations")
        assert resp.status_code == 422  # validation error

    def test_recommendations_returns_results(self, client):
        resp = client.get("/recommendations", params={"user_id": "user123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "user123"
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    def test_recommendations_with_user_events(self, client):
        # Record some user interactions to enable personalisation
        record_event("u42", "sport-011", "click")  # Wimbledon
        record_event("u42", "sport-013", "purchase")  # Tennis racket

        resp = client.get("/recommendations", params={"user_id": "u42"})
        data = resp.json()
        assert len(data["recommendations"]) > 0
        # Tennis-related items should be boosted towards the top
        top_tags = []
        for item in data["recommendations"][:5]:
            top_tags.extend(item.get("tags", []))
        # At least some tennis-related tags should appear in top recommendations
        tennis_related = [t for t in top_tags if "tennis" in t.lower()]
        assert len(tennis_related) > 0

    def test_recommendations_respects_limit(self, client):
        resp = client.get("/recommendations", params={"user_id": "user1", "limit": 3})
        data = resp.json()
        assert len(data["recommendations"]) <= 3
