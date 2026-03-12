"""Shop / merchant endpoints – create shops and list their products."""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from models.item import ItemCreate, ItemResponse
from search.engine import search_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shops", tags=["shops"])

# In-memory shop store  (shop_id → shop dict)
_shop_store: dict[str, dict] = {}


def _get_item_store() -> dict[str, dict]:
    from main import item_store
    return item_store


@router.post("", status_code=201)
async def create_shop(
    name: str = Query(..., description="Shop name"),
    description: str = Query(default="", description="Shop description"),
    location: str | None = Query(default=None, description="Shop location"),
) -> dict[str, Any]:
    """Register a new merchant shop."""
    shop_id = str(uuid.uuid4())
    shop = {
        "id": shop_id,
        "name": name,
        "description": description,
        "location": location,
        "product_ids": [],
    }
    _shop_store[shop_id] = shop
    logger.info("Created shop %s: %s", shop_id, name)
    return shop


@router.get("")
async def list_shops() -> list[dict[str, Any]]:
    """List all registered shops."""
    return list(_shop_store.values())


@router.get("/{shop_id}")
async def get_shop(shop_id: str) -> dict[str, Any]:
    """Retrieve a single shop by ID."""
    shop = _shop_store.get(shop_id)
    if shop is None:
        raise HTTPException(status_code=404, detail="Shop not found")
    return shop


@router.post("/{shop_id}/products", response_model=ItemResponse, status_code=201)
async def add_product_to_shop(shop_id: str, body: ItemCreate) -> ItemResponse:
    """Create a product listing under a specific shop."""
    shop = _shop_store.get(shop_id)
    if shop is None:
        raise HTTPException(status_code=404, detail="Shop not found")

    item_store = _get_item_store()
    item_id = str(uuid.uuid4())
    item_dict = body.model_dump()
    item_dict["id"] = item_id
    # Ensure type is set to "product" for shop listings
    item_dict.setdefault("type", "product")

    item_store[item_id] = item_dict
    search_engine.index_item(item_dict)

    # Track the product under the shop
    shop["product_ids"].append(item_id)
    logger.info("Added product %s to shop %s", item_id, shop_id)
    return ItemResponse(**item_dict)


@router.get("/{shop_id}/products", response_model=list[ItemResponse])
async def list_shop_products(shop_id: str) -> list[ItemResponse]:
    """List all products belonging to a shop."""
    shop = _shop_store.get(shop_id)
    if shop is None:
        raise HTTPException(status_code=404, detail="Shop not found")

    item_store = _get_item_store()
    products = []
    for pid in shop.get("product_ids", []):
        item = item_store.get(pid)
        if item is not None:
            products.append(ItemResponse(**item))
    return products
