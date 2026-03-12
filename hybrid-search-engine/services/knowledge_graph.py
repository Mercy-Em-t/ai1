"""Knowledge Graph: structured entity relationships across domains.

Defines typed nodes (item, category, location, query) and weighted edges
(related_to, has_event, recommended_for, co_occurs_with) to power
cross-domain reasoning and intelligent query expansion.

Populated from:
 1. Existing items (on startup)
 2. Co-clicks / co-searches (from user behaviour)
 3. Domain knowledge (static seed relationships)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ── Node types ───────────────────────────────────────────────────────────

NODE_TYPES = {"item", "category", "location", "query", "type"}

# ── Edge types ───────────────────────────────────────────────────────────

EDGE_TYPES = {"related_to", "has_event", "recommended_for", "co_occurs_with",
              "belongs_to", "located_in", "has_type"}

# ── Graph storage ────────────────────────────────────────────────────────
# node_id → { "type": str, "label": str, "meta": dict }
_nodes: dict[str, dict[str, Any]] = {}

# (source_id, target_id) → { "edge_type": str, "weight": float }
_edges: dict[tuple[str, str], dict[str, Any]] = {}

# Adjacency list: node_id → set of neighbor node_ids
_adjacency: dict[str, set[str]] = defaultdict(set)

# Co-occurrence tracker for items: item_id → set of co-interacted item_ids
_item_cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


# ── Node operations ──────────────────────────────────────────────────────


def add_node(node_id: str, node_type: str, label: str, **meta: Any) -> None:
    """Add or update a node in the knowledge graph."""
    _nodes[node_id] = {"type": node_type, "label": label, "meta": meta}


def get_node(node_id: str) -> dict[str, Any] | None:
    """Return a node or None if it doesn't exist."""
    return _nodes.get(node_id)


def node_count() -> int:
    """Return the total number of nodes."""
    return len(_nodes)


# ── Edge operations ──────────────────────────────────────────────────────


def add_edge(source_id: str, target_id: str, edge_type: str, weight: float = 1.0) -> None:
    """Add or strengthen an edge between two nodes."""
    key = (source_id, target_id)
    existing = _edges.get(key)
    if existing:
        existing["weight"] += weight
    else:
        _edges[key] = {"edge_type": edge_type, "weight": weight}
    _adjacency[source_id].add(target_id)
    _adjacency[target_id].add(source_id)


def get_edge(source_id: str, target_id: str) -> dict[str, Any] | None:
    """Return an edge or None."""
    return _edges.get((source_id, target_id)) or _edges.get((target_id, source_id))


def edge_count() -> int:
    """Return the total number of edges."""
    return len(_edges)


# ── Traversal ────────────────────────────────────────────────────────────


def get_neighbors(
    node_id: str,
    edge_type: str | None = None,
    node_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return neighbours of a node, optionally filtered by edge/node type.

    Results are sorted by edge weight (descending).
    """
    neighbor_ids = _adjacency.get(node_id, set())
    results: list[dict[str, Any]] = []

    for nid in neighbor_ids:
        edge = _edges.get((node_id, nid)) or _edges.get((nid, node_id))
        if edge is None:
            continue
        if edge_type and edge["edge_type"] != edge_type:
            continue
        node = _nodes.get(nid)
        if node is None:
            continue
        if node_type and node["type"] != node_type:
            continue
        results.append({
            "node_id": nid,
            "node_type": node["type"],
            "label": node["label"],
            "edge_type": edge["edge_type"],
            "weight": round(edge["weight"], 4),
        })

    results.sort(key=lambda x: x["weight"], reverse=True)
    return results[:limit]


def find_path(source_id: str, target_id: str, max_depth: int = 3) -> list[str] | None:
    """BFS to find the shortest path between two nodes (max depth)."""
    if source_id == target_id:
        return [source_id]
    if source_id not in _adjacency or target_id not in _adjacency:
        return None

    visited: set[str] = {source_id}
    queue: list[tuple[str, list[str]]] = [(source_id, [source_id])]

    while queue:
        current, path = queue.pop(0)
        if len(path) > max_depth:
            break
        for neighbor in _adjacency.get(current, set()):
            if neighbor == target_id:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


# ── Query expansion via KG ───────────────────────────────────────────────


def expand_query_via_kg(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Use the knowledge graph to find items related to a query.

    Walks the graph from the query node (or matching category/item nodes)
    to find related items for candidate expansion.

    Returns a list of item node_ids with relevance weights.
    """
    q_lower = query.lower().strip()
    expansions: dict[str, float] = {}

    # Direct query node lookup
    query_node_id = f"query:{q_lower}"
    if query_node_id in _nodes:
        for neighbor in get_neighbors(query_node_id, limit=20):
            if neighbor["node_type"] == "item":
                expansions[neighbor["node_id"]] = max(
                    expansions.get(neighbor["node_id"], 0.0),
                    neighbor["weight"] * 0.3,  # lower weight for KG suggestions
                )

    # Category node lookup
    cat_node_id = f"cat:{q_lower}"
    if cat_node_id in _nodes:
        for neighbor in get_neighbors(cat_node_id, limit=30):
            if neighbor["node_type"] == "item":
                expansions[neighbor["node_id"]] = max(
                    expansions.get(neighbor["node_id"], 0.0),
                    neighbor["weight"] * 0.25,
                )
            elif neighbor["node_type"] == "category":
                # Follow to items in related categories
                for item_neighbor in get_neighbors(neighbor["node_id"], node_type="item", limit=10):
                    expansions[item_neighbor["node_id"]] = max(
                        expansions.get(item_neighbor["node_id"], 0.0),
                        item_neighbor["weight"] * 0.15,
                    )

    # Look for items whose title/tags match the query terms
    for nid, node in _nodes.items():
        if node["type"] == "item":
            label_lower = node["label"].lower()
            if q_lower in label_lower:
                # Find co-occurring items
                for neighbor in get_neighbors(nid, edge_type="co_occurs_with", limit=5):
                    expansions[neighbor["node_id"]] = max(
                        expansions.get(neighbor["node_id"], 0.0),
                        neighbor["weight"] * 0.2,
                    )

    sorted_expansions = sorted(expansions.items(), key=lambda x: x[1], reverse=True)
    return [
        {"item_id": item_id.replace("item:", ""), "kg_weight": round(w, 4)}
        for item_id, w in sorted_expansions[:limit]
    ]


# ── Populate from items ──────────────────────────────────────────────────

_populated = False


def populate_from_items(item_store: dict[str, dict]) -> None:
    """Build the knowledge graph from the item store.

    Creates nodes for items, categories, locations, and types,
    and connects them with appropriate edges.
    """
    global _populated
    if _populated:
        return

    for item_id, item in item_store.items():
        title = item.get("title", "")
        category = (item.get("category") or "").lower()
        location = (item.get("location") or "").strip()
        item_type = (item.get("type") or "").lower()
        tags = item.get("tags", [])

        # Item node
        add_node(f"item:{item_id}", "item", title, category=category,
                 item_type=item_type)

        # Category node + edge
        if category:
            cat_nid = f"cat:{category}"
            add_node(cat_nid, "category", category)
            add_edge(f"item:{item_id}", cat_nid, "belongs_to", weight=1.0)

        # Location node + edge
        if location:
            loc_nid = f"loc:{location.lower()}"
            add_node(loc_nid, "location", location)
            add_edge(f"item:{item_id}", loc_nid, "located_in", weight=1.0)

        # Type node + edge
        if item_type:
            type_nid = f"type:{item_type}"
            add_node(type_nid, "type", item_type)
            add_edge(f"item:{item_id}", type_nid, "has_type", weight=1.0)

        # Tag → category edges (tags often reveal cross-domain links)
        for tag in tags:
            tag_lower = tag.lower()
            tag_nid = f"cat:{tag_lower}"
            if tag_nid not in _nodes:
                add_node(tag_nid, "category", tag_lower)
            add_edge(f"item:{item_id}", tag_nid, "related_to", weight=0.5)

    # ── Domain knowledge: inter-category relationships ────────────────
    _DOMAIN_RELATIONSHIPS = {
        "tennis": ["padel", "squash", "badminton", "sports"],
        "padel": ["tennis", "squash", "sports"],
        "football": ["sports", "basketball"],
        "basketball": ["sports", "football"],
        "sports": ["wellness", "fitness"],
        "wellness": ["fitness", "sports", "food-drink"],
        "electronics": ["technology"],
        "education": ["courses"],
        "concerts": ["festivals", "music"],
        "festivals": ["concerts", "music"],
        "food-drink": ["wellness", "services"],
    }

    for cat, related_cats in _DOMAIN_RELATIONSHIPS.items():
        cat_nid = f"cat:{cat}"
        if cat_nid not in _nodes:
            add_node(cat_nid, "category", cat)
        for rel_cat in related_cats:
            rel_nid = f"cat:{rel_cat}"
            if rel_nid not in _nodes:
                add_node(rel_nid, "category", rel_cat)
            add_edge(cat_nid, rel_nid, "related_to", weight=0.8)

    _populated = True
    logger.info(
        "Knowledge graph populated: %d nodes, %d edges",
        node_count(), edge_count(),
    )


# ── Learn from user behaviour ────────────────────────────────────────────


def record_co_interaction(item_id_a: str, item_id_b: str) -> None:
    """Record that two items were interacted with in the same session.

    Strengthens co_occurs_with edges between items.
    """
    if item_id_a == item_id_b:
        return

    nid_a = f"item:{item_id_a}"
    nid_b = f"item:{item_id_b}"

    # Only link items that exist in the graph
    if nid_a not in _nodes or nid_b not in _nodes:
        return

    _item_cooccurrence[item_id_a][item_id_b] += 1
    _item_cooccurrence[item_id_b][item_id_a] += 1

    add_edge(nid_a, nid_b, "co_occurs_with", weight=0.1)


def record_query_item_link(query: str, item_id: str) -> None:
    """Record a link between a search query and an item the user clicked."""
    q_lower = query.lower().strip()
    query_nid = f"query:{q_lower}"
    item_nid = f"item:{item_id}"

    if query_nid not in _nodes:
        add_node(query_nid, "query", q_lower)

    if item_nid in _nodes:
        add_edge(query_nid, item_nid, "recommended_for", weight=0.2)


# ── Stats ────────────────────────────────────────────────────────────────


def get_kg_stats() -> dict[str, Any]:
    """Return summary statistics for the knowledge graph."""
    type_counts: dict[str, int] = defaultdict(int)
    for node in _nodes.values():
        type_counts[node["type"]] += 1

    edge_type_counts: dict[str, int] = defaultdict(int)
    for edge in _edges.values():
        edge_type_counts[edge["edge_type"]] += 1

    return {
        "total_nodes": node_count(),
        "total_edges": edge_count(),
        "node_types": dict(type_counts),
        "edge_types": dict(edge_type_counts),
        "co_occurrence_pairs": sum(
            len(v) for v in _item_cooccurrence.values()
        ) // 2,
    }


# ── Cleanup ──────────────────────────────────────────────────────────────


def clear_knowledge_graph() -> None:
    """Clear all knowledge graph data (used in testing)."""
    global _populated
    _nodes.clear()
    _edges.clear()
    _adjacency.clear()
    _item_cooccurrence.clear()
    _populated = False
