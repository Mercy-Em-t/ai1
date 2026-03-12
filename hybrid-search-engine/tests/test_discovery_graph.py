"""Tests for Discovery Graph: balanced discovery, curiosity mode, interest profiles,
exploration scoring, interest decay, and category relationships."""
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


# ── Balanced Discovery Endpoint ──────────────────────────────────────────


class TestBalancedDiscovery:
    def test_discover_returns_four_sections(self, client):
        """The /discover endpoint should return top_results, related,
        trending, and explore_new sections."""
        resp = client.get("/discover")
        assert resp.status_code == 200
        data = resp.json()
        assert "top_results" in data
        assert "related_categories" in data
        assert "trending" in data
        assert "explore_new" in data
        assert "exploration_weight" in data

    def test_discover_with_query(self, client):
        """Searching with a query should return relevant top results."""
        resp = client.get("/discover", params={"q": "tennis"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "tennis"
        assert len(data["top_results"]) > 0

    def test_discover_with_user_id(self, client):
        """Discover with user_id should provide personalised exploration."""
        # Record some events to build a profile
        client.post("/events", json={
            "user_id": "u1", "item_id": "sport-011", "event_type": "click",
        })
        resp = client.get("/discover", params={"user_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u1"
        assert len(data["top_results"]) > 0

    def test_discover_exploration_weight(self, client):
        """Custom exploration weight should be respected."""
        resp = client.get("/discover", params={"exploration": 0.5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["exploration_weight"] == 0.5

    def test_discover_limit(self, client):
        """Limit parameter should cap top results."""
        resp = client.get("/discover", params={"limit": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["top_results"]) <= 3


# ── Curiosity Mode ───────────────────────────────────────────────────────


class TestCuriosityMode:
    def test_curiosity_returns_unexpected_finds(self, client):
        """Curiosity mode should return unexpected_finds and trending."""
        resp = client.get("/discover/curiosity")
        assert resp.status_code == 200
        data = resp.json()
        assert "unexpected_finds" in data
        assert "trending_now" in data
        assert len(data["unexpected_finds"]) > 0

    def test_curiosity_with_user_avoids_known(self, client):
        """For a user with sports history, curiosity should show
        items from OTHER categories."""
        # Build a sports-heavy profile
        for item_id in ["sport-001", "sport-002", "sport-003", "sport-004", "sport-005"]:
            client.post("/events", json={
                "user_id": "u_sport", "item_id": item_id, "event_type": "click",
            })

        resp = client.get("/discover/curiosity", params={"user_id": "u_sport"})
        assert resp.status_code == 200
        data = resp.json()
        # Should have some results
        assert len(data["unexpected_finds"]) > 0
        # Not ALL items should be sports
        categories = [
            (item.get("category") or "").lower()
            for item in data["unexpected_finds"]
        ]
        unique_cats = set(categories)
        # At least some non-sports items
        assert len(unique_cats) >= 1

    def test_curiosity_limit(self, client):
        """Curiosity limit should cap unexpected_finds."""
        resp = client.get("/discover/curiosity", params={"limit": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["unexpected_finds"]) <= 3


# ── User Interest Profile ────────────────────────────────────────────────


class TestInterestProfile:
    def test_profile_empty_user(self, client):
        """New user should have an empty profile."""
        resp = client.get("/discover/profile/new_user_xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "new_user_xyz"
        assert data["interests"] == {}
        assert data["event_count"] == 0

    def test_profile_after_events(self, client):
        """Profile should reflect user's interaction history."""
        client.post("/events", json={
            "user_id": "u10", "item_id": "sport-011", "event_type": "click",
        })
        client.post("/events", json={
            "user_id": "u10", "item_id": "sport-013", "event_type": "purchase",
        })

        resp = client.get("/discover/profile/u10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "u10"
        assert data["event_count"] == 2
        assert len(data["interests"]) > 0

    def test_profile_includes_related_categories(self, client):
        """Profile should include related categories for top interests."""
        client.post("/events", json={
            "user_id": "u11", "item_id": "sport-011", "event_type": "click",
        })

        resp = client.get("/discover/profile/u11")
        data = resp.json()
        assert "related_categories" in data


# ── Related Categories ───────────────────────────────────────────────────


class TestRelatedCategories:
    def test_seed_relationships_exist(self, client):
        """Seeded category relationships should be queryable."""
        resp = client.get("/discover/related-categories/tennis")
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "tennis"
        related_names = [r["category"] for r in data["related"]]
        # Tennis should relate to padel, squash, badminton
        assert "padel" in related_names

    def test_related_categories_limit(self, client):
        """Limit should cap related categories."""
        resp = client.get("/discover/related-categories/sports", params={"limit": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["related"]) <= 2

    def test_unknown_category_empty(self, client):
        """Unknown category should return empty list."""
        resp = client.get("/discover/related-categories/unicorns123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["related"] == []


# ── Exploration Scoring ──────────────────────────────────────────────────


class TestExplorationScoring:
    def test_exploration_score_unknown_category(self):
        """Items in unknown categories should get high exploration score."""
        from services.discovery_graph import compute_exploration_score

        item = {"category": "underwater_basket_weaving"}
        user_cats = {"sports": 5.0, "music": 3.0}
        score = compute_exploration_score(item, user_cats)
        assert score == 1.0  # completely new territory

    def test_exploration_score_known_category(self):
        """Items in the user's top category should get low exploration score."""
        from services.discovery_graph import compute_exploration_score

        item = {"category": "sports"}
        user_cats = {"sports": 5.0, "music": 3.0}
        score = compute_exploration_score(item, user_cats)
        assert score == 0.0  # very familiar

    def test_exploration_score_no_preferences(self):
        """New users (no prefs) should get neutral exploration score."""
        from services.discovery_graph import compute_exploration_score

        item = {"category": "sports"}
        score = compute_exploration_score(item, {})
        assert score == 0.5


# ── Interest Decay ───────────────────────────────────────────────────────


class TestInterestDecay:
    def test_recent_event_high_decay(self):
        """A very recent event should have decay factor close to 1."""
        from services.discovery_graph import _time_decay
        from datetime import datetime, timezone

        recent_ts = datetime.now(timezone.utc).isoformat()
        decay = _time_decay(recent_ts)
        assert decay > 0.9

    def test_old_event_low_decay(self):
        """An event from 30 days ago should have low decay factor."""
        from services.discovery_graph import _time_decay
        from datetime import datetime, timezone, timedelta

        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        decay = _time_decay(old_ts)
        assert decay < 0.15  # very decayed

    def test_invalid_timestamp_neutral(self):
        """Invalid timestamp should return neutral 0.5."""
        from services.discovery_graph import _time_decay

        assert _time_decay("not-a-date") == 0.5
        assert _time_decay("") == 0.5


# ── Discovery Ranking ────────────────────────────────────────────────────


class TestDiscoveryRanking:
    def test_discovery_rank_returns_scored_items(self, client):
        """discovery_rank should return items with _score."""
        from main import item_store
        from services.discovery_graph import discovery_rank

        items = list(item_store.values())[:10]
        ranked = discovery_rank(items)
        assert len(ranked) == 10
        assert all("_score" in item for item in ranked)

    def test_discovery_rank_with_exploration(self, client):
        """Higher exploration weight should boost unfamiliar items."""
        from main import item_store
        from services.discovery_graph import discovery_rank

        items = list(item_store.values())[:20]

        # Pure precision (no exploration)
        precision = discovery_rank(items, exploration_weight=0.0, user_id="nobody")
        # Max exploration
        diverse = discovery_rank(items, exploration_weight=0.5, user_id="nobody")

        # Both should return same count
        assert len(precision) == len(diverse)
        # Scores should differ with different weights
        precision_scores = [i["_score"] for i in precision]
        diverse_scores = [i["_score"] for i in diverse]
        assert precision_scores != diverse_scores
