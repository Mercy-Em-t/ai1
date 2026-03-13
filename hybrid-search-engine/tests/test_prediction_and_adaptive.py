"""Tests for intent prediction, auto KG growth, and adaptive exploration."""
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


# ── Intent Prediction: Transition Probabilities ─────────────────────────


class TestTransitionProbabilities:
    def test_transitions_endpoint_exists(self, client):
        """The transitions endpoint should exist."""
        resp = client.get("/predictions/transitions/query", params={"q": "padel"})
        assert resp.status_code == 200

    def test_transitions_empty_for_unknown_query(self, client):
        """Unknown query should return empty transitions."""
        resp = client.get("/predictions/transitions/query", params={"q": "xyznonexistent"})
        data = resp.json()
        assert data["transitions"] == []
        assert data["total"] == 0

    def test_transitions_after_search_sequence(self, client):
        """After a sequence of searches, transitions should appear."""
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

        resp = client.get("/predictions/transitions/query", params={"q": "padel"})
        data = resp.json()
        assert data["total"] > 0
        # Check probabilities sum to ≤ 1
        total_prob = sum(t["probability"] for t in data["transitions"])
        assert total_prob <= 1.01  # small float tolerance

    def test_transitions_have_probability_field(self, client):
        """Each transition should have a probability field."""
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "tennis",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "tennis racket",
        })

        resp = client.get("/predictions/transitions/query", params={"q": "tennis"})
        data = resp.json()
        for t in data["transitions"]:
            assert "query" in t
            assert "probability" in t
            assert 0 <= t["probability"] <= 1.0


# ── Intent Prediction: User Interest Predictions ────────────────────────


class TestUserPredictions:
    def test_predictions_endpoint_exists(self, client):
        """The predictions endpoint should exist."""
        resp = client.get("/predictions/u1")
        assert resp.status_code == 200

    def test_predictions_empty_for_new_user(self, client):
        """A user with no history should have no predictions."""
        resp = client.get("/predictions/newuser123")
        data = resp.json()
        assert data["user_id"] == "newuser123"
        assert data["predictions"] == []

    def test_predictions_after_behavior(self, client):
        """After search behavior, predictions should appear."""
        # Build query graph: padel → padel racket → padel shoes
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel racket",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "padel shoes",
        })

        resp = client.get("/predictions/u1")
        data = resp.json()
        assert data["user_id"] == "u1"
        # Should have some predictions
        assert data["total"] > 0

    def test_predictions_have_score_and_source(self, client):
        """Each prediction should have score and source fields."""
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "hiking",
        })
        client.post("/events", json={
            "user_id": "u1", "event_type": "search", "query": "camping",
        })

        resp = client.get("/predictions/u1")
        data = resp.json()
        for p in data["predictions"]:
            assert "interest" in p
            assert "score" in p
            assert "source" in p

    def test_predictions_limit_parameter(self, client):
        """The limit parameter should restrict prediction count."""
        # Build enough data for multiple predictions
        for q in ["sports", "tennis", "padel", "squash", "fitness"]:
            client.post("/events", json={
                "user_id": "u1", "event_type": "search", "query": q,
            })

        resp = client.get("/predictions/u1", params={"limit": 2})
        data = resp.json()
        assert len(data["predictions"]) <= 2


# ── Auto KG Growth: Item Similarity ─────────────────────────────────────


class TestAutoKGItemSimilarity:
    def test_kg_has_item_similarity_edges(self, client):
        """Items sharing tags should have related_to edges in the KG."""
        from services.knowledge_graph import get_kg_stats

        stats = get_kg_stats()
        edge_types = stats["edge_types"]
        # related_to edges should exist (from tags and domain knowledge)
        assert "related_to" in edge_types
        assert edge_types["related_to"] > 0

    def test_items_sharing_tags_are_connected(self, client):
        """Items with the same tag should be neighbors in the KG."""
        from services.knowledge_graph import get_neighbors

        # Find two items that share a tag by checking neighbors
        # sport-001 is a football item with "football" tag
        neighbors = get_neighbors("item:sport-001", edge_type="related_to", limit=50)
        # Should have at least tag-based connections
        assert len(neighbors) > 0

    def test_kg_stats_show_edges(self, client):
        """KG stats should reflect the item similarity edges."""
        resp = client.get("/knowledge-graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_edges"] > 0
        assert data["total_nodes"] > 0


# ── Adaptive Exploration ────────────────────────────────────────────────


class TestAdaptiveExploration:
    def test_new_user_gets_higher_exploration(self, client):
        """A new user (no history) should get higher exploration weight."""
        from services.discovery_graph import compute_adaptive_exploration

        weight = compute_adaptive_exploration(user_id=None)
        assert weight > 0.20  # higher than default

    def test_exact_query_gets_low_exploration(self, client):
        """An exact query match should result in low exploration."""
        from services.discovery_graph import compute_adaptive_exploration

        weight = compute_adaptive_exploration(user_id="u1", has_exact_query=True)
        assert weight <= 0.15

    def test_heavy_user_gets_lower_exploration(self, client):
        """A user with lots of history should get lower exploration."""
        from services.discovery_graph import compute_adaptive_exploration

        # Build a heavy user
        for i in range(15):
            client.post("/events", json={
                "user_id": "heavy_user",
                "item_id": f"prod-00{(i % 5) + 1}",
                "event_type": "click",
            })

        weight = compute_adaptive_exploration(user_id="heavy_user")
        assert weight < 0.25

    def test_discover_endpoint_uses_adaptive_weight(self, client):
        """The discover endpoint should return adaptive exploration weight."""
        # Anonymous user, no query — should get high exploration
        resp = client.get("/discover")
        data = resp.json()
        assert "exploration_weight" in data
        # Anonymous browsing → high exploration
        assert data["exploration_weight"] >= 0.20

    def test_discover_with_query_lower_exploration(self, client):
        """Discover with a query should have lower exploration weight."""
        resp = client.get("/discover", params={"q": "tennis"})
        data = resp.json()
        # Query present → lower exploration than browsing
        assert data["exploration_weight"] <= 0.20

    def test_discover_explicit_override_respected(self, client):
        """Explicit exploration weight should override adaptive logic."""
        resp = client.get("/discover", params={"exploration": 0.50})
        data = resp.json()
        assert data["exploration_weight"] == 0.50
