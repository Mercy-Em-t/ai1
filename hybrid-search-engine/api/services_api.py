"""Service listing endpoints – create and browse services."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from models.item import ItemCreate, ItemResponse
from search.engine import search_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["services"])


def _get_store() -> dict[str, dict]:
    from main import item_store
    return item_store


@router.post("", response_model=ItemResponse, status_code=201)
async def create_service(body: ItemCreate) -> ItemResponse:
    """Create a new service listing."""
    store = _get_store()
    item_id = str(uuid.uuid4())
    item_dict = body.model_dump()
    item_dict["id"] = item_id
    item_dict["type"] = "service"  # Always set type to service
    store[item_id] = item_dict
    search_engine.index_item(item_dict)
    logger.info("Created service %s", item_id)
    return ItemResponse(**item_dict)


@router.get("", response_model=list[ItemResponse])
async def list_services(
    category: str | None = Query(default=None, description="Filter by category"),
    location: str | None = Query(default=None, description="Filter by location (substring match)"),
) -> list[ItemResponse]:
    """List all service listings, optionally filtered."""
    store = _get_store()
    services = [i for i in store.values() if i.get("type") == "service"]
    if category:
        services = [i for i in services if (i.get("category") or "").lower() == category.lower()]
    if location:
        loc_lower = location.lower()
        services = [
            i for i in services
            if (i.get("location") or "").lower().find(loc_lower) >= 0
        ]
    return [ItemResponse(**i) for i in services]


@router.get("/{service_id}", response_model=ItemResponse)
async def get_service(service_id: str) -> ItemResponse:
    """Retrieve a single service by ID."""
    store = _get_store()
    item = store.get(service_id)
    if item is None or item.get("type") != "service":
        raise HTTPException(status_code=404, detail="Service not found")
    return ItemResponse(**item)
