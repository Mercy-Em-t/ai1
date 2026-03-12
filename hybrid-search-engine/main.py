"""FastAPI entry point – loads sample data and wires all routers."""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Global in-memory item store  (item_id → item dict)
item_store: dict[str, dict] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load sample data on startup."""
    from data.sample_data import SAMPLE_ITEMS
    from search.engine import search_engine

    logger.info("Loading %d sample items into store and search index …", len(SAMPLE_ITEMS))
    for raw in SAMPLE_ITEMS:
        item = dict(raw)
        item.setdefault("id", str(uuid.uuid4()))
        item_store[item["id"]] = item
        search_engine.index_item(item)

    logger.info(
        "Startup complete – %d items indexed, vocabulary size=%d",
        search_engine.document_count,
        len(search_engine.vocabulary),
    )
    yield
    logger.info("Shutdown – clearing store.")
    item_store.clear()


app = FastAPI(
    title="Hybrid Search Engine",
    description="A full hybrid search pipeline: TF-IDF + personalisation + filtering + ranking.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
from api.items import router as items_router
from api.search import router as search_router
from api.events import router as events_router

app.include_router(items_router)
app.include_router(search_router)
app.include_router(events_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok", "service": "hybrid-search-engine"}
