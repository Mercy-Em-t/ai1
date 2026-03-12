"""Knowledge Graph API: entity traversal, query expansion, and graph statistics."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query

from services.knowledge_graph import (
    expand_query_via_kg,
    find_path,
    get_kg_stats,
    get_neighbors,
    get_node,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


@router.get("/node/{node_id:path}")
async def get_node_endpoint(
    node_id: str,
    neighbor_limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Return a node and its neighbours in the knowledge graph.

    Use prefixes to address nodes:
    - ``item:<item_id>``
    - ``cat:<category>``
    - ``loc:<location>``
    - ``type:<type>``
    - ``query:<query>``
    """
    node = get_node(node_id)
    if node is None:
        return {"node_id": node_id, "found": False}

    neighbors = get_neighbors(node_id, limit=neighbor_limit)
    return {
        "node_id": node_id,
        "found": True,
        "node_type": node["type"],
        "label": node["label"],
        "neighbors": neighbors,
        "neighbor_count": len(neighbors),
    }


@router.get("/expand")
async def expand_query_endpoint(
    q: str = Query(..., description="Query to expand via the knowledge graph"),
    limit: int = Query(default=10, ge=1, le=50),
) -> dict[str, Any]:
    """Use the knowledge graph to expand a query and find related items.

    Returns item IDs with relevance weights derived from graph traversal.
    KG suggestions carry lower weight to maintain user control over results.
    """
    expansions = expand_query_via_kg(q, limit=limit)

    # Resolve item titles for readability
    from main import item_store
    enriched = []
    for exp in expansions:
        item = item_store.get(exp["item_id"])
        enriched.append({
            **exp,
            "title": item["title"] if item else None,
            "category": item.get("category") if item else None,
        })

    return {
        "query": q,
        "expansions": enriched,
        "total": len(enriched),
    }


@router.get("/path")
async def find_path_endpoint(
    source: str = Query(..., description="Source node ID (e.g. cat:tennis)"),
    target: str = Query(..., description="Target node ID (e.g. cat:wellness)"),
    max_depth: int = Query(default=4, ge=1, le=6),
) -> dict[str, Any]:
    """Find the shortest path between two nodes in the knowledge graph."""
    path = find_path(source, target, max_depth=max_depth)
    if path is None:
        return {"source": source, "target": target, "found": False, "path": []}

    # Resolve labels
    resolved = []
    for nid in path:
        node = get_node(nid)
        resolved.append({
            "node_id": nid,
            "label": node["label"] if node else nid,
            "node_type": node["type"] if node else "unknown",
        })

    return {
        "source": source,
        "target": target,
        "found": True,
        "path": resolved,
        "length": len(resolved),
    }


@router.get("/stats")
async def kg_stats_endpoint() -> dict[str, Any]:
    """Return summary statistics for the knowledge graph."""
    return get_kg_stats()
