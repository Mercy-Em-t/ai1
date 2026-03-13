"""Tests for graceful zero-result recovery and unmet demand capture."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from services.analytics import clear_logs
from services.cache import _memory_cache
from services.intelligence import (
    clear_intelligence,
    get_unmet_demand,
    record_unmet_demand,
)
from services.discovery_graph import clear_discovery_graph
from services.knowledge_graph import clear_knowledge_graph


@pytest.fixture()
def client():
    """Create a TestClient with lifespan so sample data is loaded."""
    clear_logs()
    _memory_cache.clear()
    clear_intelligence()
    clear_discovery_graph()
    with TestClient(app) as c:
        yield c
    clear_logs()
    clear_intelligence()
    clear_discovery_graph()
    clear_knowledge_graph()


# ── Graceful Zero-Result Recovery ────────────────────────────────────────


class TestZeroResultRecovery:
    def test_search_response_has_suggestions_field(self, client):
        """All search responses should include suggestions field."""
        resp = client.get("/search", params={"q": "tennis"})
        assert resp.status_code == 200
        data = resp.json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_search_response_has_fallback_results_field(self, client):
        """All search responses should include fallback_results field."""
        resp = client.get("/search", params={"q": "tennis"})
        assert resp.status_code == 200
        data = resp.json()
        assert "fallback_results" in data
        assert isinstance(data["fallback_results"], list)

    def test_normal_search_has_empty_fallback(self, client):
        """When results are found, fallback_results and suggestions are empty."""
        resp = client.get("/search", params={"q": "tennis"})
        data = resp.json()
        assert data["total"] > 0
        assert data["fallback_results"] == []
        assert data["suggestions"] == []

    def test_zero_result_search_returns_fallback(self, client):
        """Filtering to force zero results should trigger fallback recovery."""
        # Search for tennis (exists) but with an impossible location
        resp = client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["results"] == []
        # Recovery should provide fallback results since tennis is a known topic
        assert len(data["fallback_results"]) > 0
        assert len(data["suggestions"]) > 0

    def test_fallback_results_have_score(self, client):
        """Fallback result items should include a score field."""
        resp = client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })
        data = resp.json()
        assert data["total"] == 0
        for item in data["fallback_results"]:
            assert "score" in item

    def test_suggestions_are_strings(self, client):
        """Suggestions should be a list of strings."""
        resp = client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })
        data = resp.json()
        for suggestion in data["suggestions"]:
            assert isinstance(suggestion, str)

    def test_suggestions_are_limited(self, client):
        """At most 5 suggestions should be returned."""
        resp = client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })
        data = resp.json()
        assert len(data["suggestions"]) <= 5


# ── Unmet Demand Tracking (Service Level) ────────────────────────────────


class TestUnmetDemandService:
    def test_record_and_retrieve_demand(self, client):
        """Recording demand should make it retrievable."""
        record_unmet_demand("padel shoes")
        record_unmet_demand("padel shoes")
        record_unmet_demand("climbing wall")

        demand = get_unmet_demand(limit=10)
        assert len(demand) == 2
        assert demand[0]["query"] == "padel shoes"
        assert demand[0]["search_count"] == 2
        assert demand[1]["query"] == "climbing wall"
        assert demand[1]["search_count"] == 1

    def test_demand_cleared_on_clear(self, client):
        """clear_intelligence should also clear demand data."""
        record_unmet_demand("test query")
        clear_intelligence()
        assert get_unmet_demand() == []

    def test_empty_query_not_recorded(self, client):
        """Empty or whitespace queries should not be recorded."""
        record_unmet_demand("")
        record_unmet_demand("   ")
        assert get_unmet_demand() == []

    def test_demand_case_insensitive(self, client):
        """Demand tracking should normalize queries to lowercase."""
        record_unmet_demand("Padel Shoes")
        record_unmet_demand("padel shoes")
        demand = get_unmet_demand()
        assert len(demand) == 1
        assert demand[0]["search_count"] == 2


# ── Unmet Demand API Endpoint ────────────────────────────────────────────


class TestUnmetDemandEndpoint:
    def test_demand_endpoint_exists(self, client):
        """GET /intelligence/demand should return 200."""
        resp = client.get("/intelligence/demand")
        assert resp.status_code == 200

    def test_demand_endpoint_empty_initially(self, client):
        """Demand endpoint returns empty list when no zero-result searches."""
        resp = client.get("/intelligence/demand")
        data = resp.json()
        assert data["unmet_demand"] == []
        assert data["total"] == 0

    def test_demand_endpoint_after_zero_result_search(self, client):
        """Zero-result searches should appear in the demand endpoint."""
        # Force a zero-result search by using impossible location filter
        client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })

        resp = client.get("/intelligence/demand")
        data = resp.json()
        assert data["total"] > 0
        demand_queries = [d["query"] for d in data["unmet_demand"]]
        assert "tennis" in demand_queries

    def test_demand_endpoint_limit_parameter(self, client):
        """Demand endpoint respects the limit parameter."""
        # Generate multiple unique zero-result searches
        for i in range(5):
            _memory_cache.clear()
            client.get("/search", params={
                "q": f"query{i}",
                "location": "xyznonexistentplace",
            })

        resp = client.get("/intelligence/demand", params={"limit": 3})
        data = resp.json()
        assert len(data["unmet_demand"]) <= 3

    def test_successful_search_not_in_demand(self, client):
        """Searches with results should not appear in unmet demand."""
        # First, perform a zero-result search to ensure tracking is active
        client.get("/search", params={
            "q": "forcedzeroresult",
            "location": "xyznonexistentplace",
        })
        # Now perform a successful search
        client.get("/search", params={"q": "tennis"})

        resp = client.get("/intelligence/demand")
        data = resp.json()
        # The zero-result query should be tracked
        demand_queries = [d["query"] for d in data["unmet_demand"]]
        assert "forcedzeroresult" in demand_queries
        # But the successful query should not
        assert "tennis" not in demand_queries

    def test_dashboard_includes_unmet_demand(self, client):
        """Intelligence dashboard should include unmet_demand field."""
        client.get("/search", params={
            "q": "tennis",
            "location": "xyznonexistentplace",
        })

        resp = client.get("/intelligence/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "unmet_demand" in data
        assert isinstance(data["unmet_demand"], list)
        assert len(data["unmet_demand"]) > 0

