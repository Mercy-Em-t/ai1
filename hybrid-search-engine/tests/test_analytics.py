"""Tests for search analytics logging."""
from __future__ import annotations

import sys
import os

# Ensure the hybrid-search-engine root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.analytics import clear_logs, get_popular_queries, get_search_logs, log_search


class TestLogSearch:
    def setup_method(self):
        clear_logs()

    def test_log_search_basic(self):
        log_search(
            query="tennis racket",
            corrected_query=None,
            domain="sports",
            results_count=5,
            query_time_ms=12.5,
            user_id="user123",
        )
        logs = get_search_logs()
        assert len(logs) == 1
        assert logs[0]["query"] == "tennis racket"
        assert logs[0]["corrected_query"] is None
        assert logs[0]["domain"] == "sports"
        assert logs[0]["results_count"] == 5
        assert logs[0]["query_time_ms"] == 12.5
        assert logs[0]["user_id"] == "user123"
        assert "timestamp" in logs[0]

    def test_log_search_without_user_id(self):
        log_search(
            query="headphones",
            corrected_query=None,
            domain="products",
            results_count=3,
            query_time_ms=8.2,
        )
        logs = get_search_logs()
        assert len(logs) == 1
        assert logs[0]["user_id"] is None

    def test_log_search_with_correction(self):
        log_search(
            query="tenns rackt",
            corrected_query="tennis racket",
            domain="sports",
            results_count=5,
            query_time_ms=15.0,
        )
        logs = get_search_logs()
        assert logs[0]["corrected_query"] == "tennis racket"

    def test_zero_results_tracking(self):
        log_search(query="xyz123", corrected_query=None, domain="products",
                   results_count=0, query_time_ms=2.0)
        log_search(query="tennis", corrected_query=None, domain="sports",
                   results_count=10, query_time_ms=5.0)
        log_search(query="qwerty", corrected_query=None, domain="products",
                   results_count=0, query_time_ms=3.0)

        zero_result_logs = get_search_logs(zero_results_only=True)
        assert len(zero_result_logs) == 2
        assert zero_result_logs[0]["query"] == "xyz123"
        assert zero_result_logs[1]["query"] == "qwerty"

    def test_get_search_logs_limit(self):
        for i in range(20):
            log_search(query=f"q{i}", corrected_query=None, domain="products",
                       results_count=i, query_time_ms=1.0)
        logs = get_search_logs(limit=5)
        assert len(logs) == 5
        # Should return last 5
        assert logs[0]["query"] == "q15"

    def test_popular_queries(self):
        for _ in range(5):
            log_search(query="tennis", corrected_query=None, domain="sports",
                       results_count=10, query_time_ms=5.0)
        for _ in range(3):
            log_search(query="headphones", corrected_query=None, domain="products",
                       results_count=5, query_time_ms=4.0)
        log_search(query="laptop", corrected_query=None, domain="products",
                   results_count=8, query_time_ms=3.0)

        popular = get_popular_queries(limit=3)
        assert len(popular) == 3
        assert popular[0]["query"] == "tennis"
        assert popular[0]["count"] == 5
        assert popular[1]["query"] == "headphones"
        assert popular[1]["count"] == 3
        assert popular[2]["query"] == "laptop"
        assert popular[2]["count"] == 1

    def test_clear_logs(self):
        log_search(query="test", corrected_query=None, domain="products",
                   results_count=1, query_time_ms=1.0)
        assert len(get_search_logs()) == 1
        clear_logs()
        assert len(get_search_logs()) == 0
