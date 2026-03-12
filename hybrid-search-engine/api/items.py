"""CRUD endpoints for items."""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query

from models.item import ItemCreate, ItemResponse, ItemUpdate
from search.engine import search_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/items", tags=["items"])


def _get_store():
    from main import item_store
    return item_store


@router.post("", response_model=ItemResponse, status_code=201)
async def create_item(body: ItemCreate) -> ItemResponse:
    store = _get_store()
    item_id = str(uuid.uuid4())
    item_dict = body.model_dump()
    item_dict["id"] = item_id
    store[item_id] = item_dict
    search_engine.index_item(item_dict)
    logger.info("Created item %s", item_id)
    return ItemResponse(**item_dict)


@router.get("", response_model=list[ItemResponse])
async def list_items(
    category: str | None = Query(default=None),
) -> list[ItemResponse]:
    store = _get_store()
    items = list(store.values())
    if category:
        items = [i for i in items if i.get("category", "").lower() == category.lower()]
    return [ItemResponse(**i) for i in items]


@router.get("/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str) -> ItemResponse:
    store = _get_store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(**item)


@router.put("/{item_id}", response_model=ItemResponse)
async def update_item(item_id: str, body: ItemUpdate) -> ItemResponse:
    store = _get_store()
    item = store.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    updates = body.model_dump(exclude_none=True)
    item.update(updates)
    store[item_id] = item
    search_engine.index_item(item)
    logger.info("Updated item %s", item_id)
    return ItemResponse(**item)


@router.delete("/{item_id}", status_code=204)
async def delete_item(item_id: str) -> None:
    store = _get_store()
    if item_id not in store:
        raise HTTPException(status_code=404, detail="Item not found")
    del store[item_id]
    search_engine.remove_item(item_id)
    logger.info("Deleted item %s", item_id)
