"""Tests for semantic embedding service."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.embeddings import compute_item_text, cosine_similarity


class TestCosineSimiliarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(vec_a, vec_b)) < 1e-6

    def test_opposite_vectors(self):
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [-1.0, -2.0, -3.0]
        assert abs(cosine_similarity(vec_a, vec_b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_similar_vectors_high_score(self):
        vec_a = [0.23, -0.17, 0.45, 0.89]
        vec_b = [0.25, -0.16, 0.44, 0.90]
        score = cosine_similarity(vec_a, vec_b)
        assert score > 0.99  # Should be very similar


class TestComputeItemText:
    def test_full_item(self):
        item = {
            "title": "Water Bottle",
            "description": "A great water bottle",
            "category": "products",
            "tags": ["water", "bottle"],
        }
        text = compute_item_text(item)
        assert "Water Bottle" in text
        assert "A great water bottle" in text
        assert "products" in text
        assert "water" in text
        assert "bottle" in text

    def test_minimal_item(self):
        item = {"title": "Test"}
        text = compute_item_text(item)
        assert text == "Test"

    def test_empty_item(self):
        text = compute_item_text({})
        assert text == ""

    def test_item_with_empty_tags(self):
        item = {"title": "Test", "tags": []}
        text = compute_item_text(item)
        assert text == "Test"
