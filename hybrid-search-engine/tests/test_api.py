"""Integration tests for the new API endpoints using FastAPI TestClient."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from services.analytics import clear_logs
from services.cache import _memory_cache


@pytest.fixture()
def client():
    """Create a TestClient with lifespan so sample data is loaded."""
    clear_logs()
    _memory_cache.clear()
    with TestClient(app) as c:
        yield c
    clear_logs()


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestSearchEndpoint:
    def test_search_returns_results(self, client):
        resp = client.get("/search", params={"q": "tennis"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "total" in data
        assert "domain" in data
        assert "query_time_ms" in data
        assert data["total"] > 0

    def test_search_logs_analytics(self, client):
        client.get("/search", params={"q": "headphones"})
        resp = client.get("/analytics/searches")
        assert resp.status_code == 200
        logs = resp.json()
        assert len(logs) >= 1
        assert logs[-1]["query"] == "headphones"

    def test_search_with_user_id(self, client):
        client.get("/search", params={"q": "laptop", "user_id": "u1"})
        resp = client.get("/analytics/searches")
        logs = resp.json()
        assert any(log["user_id"] == "u1" for log in logs)


class TestSuggestEndpoint:
    def test_suggest_returns_list(self, client):
        resp = client.get("/suggest", params={"q": "ten"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_suggest_limit(self, client):
        resp = client.get("/suggest", params={"q": "ten", "limit": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 2

    def test_suggest_requires_q(self, client):
        resp = client.get("/suggest")
        assert resp.status_code == 422  # validation error


class TestAnalyticsEndpoint:
    def test_analytics_searches_empty(self, client):
        resp = client.get("/analytics/searches")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_analytics_popular_empty(self, client):
        resp = client.get("/analytics/popular")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_analytics_zero_results_filter(self, client):
        resp = client.get("/analytics/searches", params={"zero_results_only": True})
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_analytics_popular_after_searches(self, client):
        # Use different queries to avoid cache hits
        client.get("/search", params={"q": "tennis"})
        _memory_cache.clear()  # Clear cache between identical queries
        client.get("/search", params={"q": "tennis"})
        client.get("/search", params={"q": "laptop"})
        resp = client.get("/analytics/popular")
        data = resp.json()
        assert len(data) >= 2
        assert data[0]["query"] == "tennis"
        assert data[0]["count"] == 2
