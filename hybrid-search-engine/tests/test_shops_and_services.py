"""Tests for shops/merchant and services endpoints."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from main import app
from services.analytics import clear_logs
from services.cache import _memory_cache
from api.shops import _shop_store


@pytest.fixture()
def client():
    """Create a TestClient with lifespan so sample data is loaded."""
    clear_logs()
    _memory_cache.clear()
    _shop_store.clear()
    with TestClient(app) as c:
        yield c
    clear_logs()
    _shop_store.clear()


class TestShopsEndpoint:
    def test_create_shop(self, client):
        resp = client.post("/shops", params={"name": "My Tennis Shop", "location": "Nairobi"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "My Tennis Shop"
        assert data["location"] == "Nairobi"
        assert "id" in data
        assert data["product_ids"] == []

    def test_list_shops_empty(self, client):
        resp = client.get("/shops")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_shops_after_creation(self, client):
        client.post("/shops", params={"name": "Shop A"})
        client.post("/shops", params={"name": "Shop B"})
        resp = client.get("/shops")
        assert len(resp.json()) == 2

    def test_get_shop(self, client):
        create_resp = client.post("/shops", params={"name": "My Shop"})
        shop_id = create_resp.json()["id"]
        resp = client.get(f"/shops/{shop_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "My Shop"

    def test_get_shop_not_found(self, client):
        resp = client.get("/shops/nonexistent")
        assert resp.status_code == 404

    def test_add_product_to_shop(self, client):
        # Create a shop first
        create_resp = client.post("/shops", params={"name": "Tennis Pro"})
        shop_id = create_resp.json()["id"]

        # Add a product
        product = {
            "title": "Pro Tennis Racket",
            "description": "High-end tennis racket",
            "category": "tennis",
            "tags": ["tennis", "racket"],
            "price": 299.00,
        }
        resp = client.post(f"/shops/{shop_id}/products", json=product)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Pro Tennis Racket"
        assert data["type"] == "product"

    def test_list_shop_products(self, client):
        # Create shop and add products
        create_resp = client.post("/shops", params={"name": "Gear Store"})
        shop_id = create_resp.json()["id"]

        client.post(f"/shops/{shop_id}/products", json={
            "title": "Item A", "description": "Desc A", "category": "electronics",
        })
        client.post(f"/shops/{shop_id}/products", json={
            "title": "Item B", "description": "Desc B", "category": "electronics",
        })

        resp = client.get(f"/shops/{shop_id}/products")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_add_product_to_nonexistent_shop(self, client):
        product = {
            "title": "Orphan Product",
            "description": "No shop",
            "category": "electronics",
        }
        resp = client.post("/shops/nonexistent/products", json=product)
        assert resp.status_code == 404


class TestServicesEndpoint:
    def test_list_services(self, client):
        resp = client.get("/services")
        assert resp.status_code == 200
        data = resp.json()
        # Should include the sample service items
        assert len(data) > 0
        for item in data:
            assert item.get("type") == "service"

    def test_create_service(self, client):
        service = {
            "title": "Personal Training",
            "description": "One-on-one fitness coaching",
            "category": "wellness",
            "tags": ["fitness", "training"],
            "price": 80.00,
        }
        resp = client.post("/services", json=service)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Personal Training"
        assert data["type"] == "service"

    def test_get_service(self, client):
        resp = client.get("/services")
        services = resp.json()
        if services:
            service_id = services[0]["id"]
            resp = client.get(f"/services/{service_id}")
            assert resp.status_code == 200
            assert resp.json()["type"] == "service"

    def test_get_service_not_found(self, client):
        resp = client.get("/services/nonexistent")
        assert resp.status_code == 404

    def test_list_services_filter_by_location(self, client):
        resp = client.get("/services", params={"location": "Nairobi"})
        assert resp.status_code == 200
        for item in resp.json():
            assert "nairobi" in (item.get("location") or "").lower()

    def test_list_services_filter_by_category(self, client):
        resp = client.get("/services", params={"category": "tennis"})
        assert resp.status_code == 200
        for item in resp.json():
            assert item.get("category") == "tennis"


class TestItemsTypeFilter:
    def test_items_filter_by_type(self, client):
        resp = client.get("/items", params={"type": "venue"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        for item in data:
            assert item.get("type") == "venue"

    def test_items_filter_by_type_product(self, client):
        resp = client.get("/items", params={"type": "product"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        for item in data:
            assert item.get("type") == "product"

    def test_items_type_has_field(self, client):
        """All items should now have a type field."""
        resp = client.get("/items")
        data = resp.json()
        for item in data:
            assert item.get("type") is not None, f"Item {item['id']} missing type"
