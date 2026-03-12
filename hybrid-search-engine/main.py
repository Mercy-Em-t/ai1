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

    # ── Generate semantic embeddings for all items ────────────────────────
    try:
        from services.embeddings import compute_item_text, encode_texts

        items_needing_vectors = [
            (iid, item)
            for iid, item in item_store.items()
            if item.get("vector") is None
        ]
        if items_needing_vectors:
            texts = [compute_item_text(item) for _, item in items_needing_vectors]
            vectors = encode_texts(texts)
            if vectors is not None:
                for (iid, item), vec in zip(items_needing_vectors, vectors):
                    item["vector"] = vec
                logger.info("Generated embeddings for %d items.", len(vectors))
            else:
                logger.info("Semantic embeddings skipped (model not available).")
    except Exception:
        logger.info("Semantic embeddings skipped (sentence-transformers not installed).")

    yield
    logger.info("Shutdown – clearing store.")
    item_store.clear()


app = FastAPI(
    title="Hybrid Search Engine",
    description="Universal discovery engine: TF-IDF + semantic search + personalisation + filtering + ranking. "
    "Powers products, events, venues, services, courses, and experiences.",
    version="2.0.0",
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
from api.analytics import router as analytics_router
from api.discovery import router as discovery_router
from api.intelligence import router as intelligence_router
from api.items import router as items_router
from api.search import router as search_router
from api.services_api import router as services_router
from api.shops import router as shops_router
from api.suggest import router as suggest_router
from api.events import router as events_router

app.include_router(items_router)
app.include_router(search_router)
app.include_router(suggest_router)
app.include_router(events_router)
app.include_router(analytics_router)
app.include_router(discovery_router)
app.include_router(shops_router)
app.include_router(services_router)
app.include_router(intelligence_router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok", "service": "hybrid-search-engine"}
