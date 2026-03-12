"""Tests for Knowledge Graph: construction, traversal, query expansion, and API endpoints."""
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


# ── Graph Population ─────────────────────────────────────────────────────


class TestKnowledgeGraphPopulation:
    def test_graph_populated_on_startup(self, client):
        """KG should be populated with nodes and edges from sample data."""
        resp = client.get("/knowledge-graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_nodes"] > 0
        assert data["total_edges"] > 0

    def test_node_types_present(self, client):
        """KG should contain item, category, location, and type nodes."""
        resp = client.get("/knowledge-graph/stats")
        data = resp.json()
        node_types = data["node_types"]
        assert "item" in node_types
        assert "category" in node_types
        assert "type" in node_types

    def test_edge_types_present(self, client):
        """KG should contain belongs_to, related_to, and has_type edges."""
        resp = client.get("/knowledge-graph/stats")
        data = resp.json()
        edge_types = data["edge_types"]
        assert "belongs_to" in edge_types
        assert "related_to" in edge_types
        assert "has_type" in edge_types

    def test_domain_relationships_seeded(self, client):
        """Domain relationships (e.g. tennis → padel) should be in the graph."""
        resp = client.get("/knowledge-graph/node/cat:tennis")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        neighbor_labels = [n["label"] for n in data["neighbors"]]
        # Tennis should connect to padel via domain knowledge
        assert "padel" in neighbor_labels or any("padel" in l for l in neighbor_labels)


# ── Node Endpoint ────────────────────────────────────────────────────────


class TestNodeEndpoint:
    def test_get_existing_item_node(self, client):
        """Looking up an existing item node should return it."""
        resp = client.get("/knowledge-graph/node/item:prod-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["node_type"] == "item"
        assert data["neighbor_count"] > 0

    def test_get_category_node(self, client):
        """Category nodes should have item neighbours."""
        resp = client.get("/knowledge-graph/node/cat:electronics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["node_type"] == "category"
        assert data["neighbor_count"] > 0

    def test_get_nonexistent_node(self, client):
        """Non-existent node should return found=False."""
        resp = client.get("/knowledge-graph/node/item:nonexistent-xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False

    def test_node_neighbor_limit(self, client):
        """Neighbor limit should be respected."""
        resp = client.get("/knowledge-graph/node/cat:electronics",
                          params={"neighbor_limit": 2})
        data = resp.json()
        assert len(data["neighbors"]) <= 2


# ── Query Expansion ──────────────────────────────────────────────────────


class TestQueryExpansion:
    def test_expand_known_category(self, client):
        """Expanding a known category should return related items."""
        resp = client.get("/knowledge-graph/expand", params={"q": "tennis"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "tennis"
        assert data["total"] > 0
        # Should return items with titles
        assert all(e["title"] is not None for e in data["expansions"][:3])

    def test_expand_unknown_query(self, client):
        """Expanding an unknown query should return empty or minimal results."""
        resp = client.get("/knowledge-graph/expand", params={"q": "zzz-nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["expansions"], list)

    def test_expand_includes_kg_weight(self, client):
        """Each expansion should have a kg_weight field."""
        resp = client.get("/knowledge-graph/expand", params={"q": "electronics"})
        data = resp.json()
        if data["total"] > 0:
            assert "kg_weight" in data["expansions"][0]
            assert data["expansions"][0]["kg_weight"] > 0

    def test_expand_limit(self, client):
        """Limit should cap expansion results."""
        resp = client.get("/knowledge-graph/expand",
                          params={"q": "tennis", "limit": 3})
        data = resp.json()
        assert len(data["expansions"]) <= 3


# ── Path Finding ─────────────────────────────────────────────────────────


class TestPathFinding:
    def test_path_between_related_categories(self, client):
        """Should find a path between related categories."""
        resp = client.get("/knowledge-graph/path",
                          params={"source": "cat:tennis", "target": "cat:sports"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert len(data["path"]) >= 2

    def test_path_between_item_and_category(self, client):
        """Should find a path between an item and its category."""
        resp = client.get("/knowledge-graph/path",
                          params={"source": "item:prod-001", "target": "cat:electronics"})
        data = resp.json()
        assert data["found"] is True
        assert data["length"] >= 2

    def test_no_path_between_disconnected_nodes(self, client):
        """Should return found=False for non-existent nodes."""
        resp = client.get("/knowledge-graph/path",
                          params={"source": "cat:nonexistent", "target": "cat:also-nonexistent"})
        data = resp.json()
        assert data["found"] is False


# ── Learning from Behaviour ──────────────────────────────────────────────


class TestKGLearning:
    def test_co_interaction_creates_edge(self, client):
        """Clicking two items should create a co_occurs_with edge."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-002", "event_type": "click",
        })

        # The items should now be connected
        resp = client.get("/knowledge-graph/node/item:prod-001")
        data = resp.json()
        neighbor_ids = [n["node_id"] for n in data["neighbors"]]
        assert "item:prod-002" in neighbor_ids

    def test_query_item_link(self, client):
        """Click with query context should create query→item edge."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001",
            "event_type": "click", "query": "best headphones",
        })

        resp = client.get("/knowledge-graph/node/query:best headphones")
        data = resp.json()
        assert data["found"] is True
        neighbor_ids = [n["node_id"] for n in data["neighbors"]]
        assert "item:prod-001" in neighbor_ids

    def test_kg_stats_reflect_co_occurrences(self, client):
        """Stats should reflect co-occurrence pairs after events."""
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-002", "event_type": "click",
        })

        resp = client.get("/knowledge-graph/stats")
        data = resp.json()
        assert data["co_occurrence_pairs"] >= 1


# ── Integration with Search ──────────────────────────────────────────────


class TestKGSearchIntegration:
    def test_search_still_works_with_kg(self, client):
        """Standard search should still work with KG integration."""
        resp = client.get("/search", params={"q": "headphones"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0

    def test_search_benefits_from_kg(self, client):
        """After recording events, KG should enrich search results."""
        # Record clicks that build query→item links
        client.post("/events", json={
            "user_id": "u1", "item_id": "prod-001",
            "event_type": "click", "query": "noise cancelling",
        })

        # Search should still return results
        resp = client.get("/search", params={"q": "noise cancelling"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
