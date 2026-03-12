"""In-memory TF-IDF-like search engine with an inverted index."""
from __future__ import annotations

import logging
import math
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "it", "its", "this",
        "that", "these", "those", "i", "you", "he", "she", "we", "they",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class SearchEngine:
    """
    In-memory search engine using TF-IDF scoring with an inverted index.

    Indexed fields: title (weight 3), tags (weight 2),
                    category (weight 2), description (weight 1),
                    location (weight 1).
    """

    FIELD_WEIGHTS: dict[str, float] = {
        "title": 3.0,
        "tags": 2.0,
        "category": 2.0,
        "description": 1.0,
        "location": 1.0,
    }

    def __init__(self) -> None:
        # item_id -> Item dict
        self._documents: dict[str, dict] = {}
        # token -> {item_id: weighted_tf}
        self._inverted_index: dict[str, dict[str, float]] = defaultdict(dict)
        # item_id -> total token count (for TF normalisation)
        self._doc_lengths: dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Index management
    # ──────────────────────────────────────────────────────────────────────

    def index_item(self, item: dict) -> None:
        """Add or replace an item in the index."""
        item_id: str = item["id"]
        # Remove stale data first
        self.remove_item(item_id)

        self._documents[item_id] = item
        token_weights: dict[str, float] = defaultdict(float)

        for field, weight in self.FIELD_WEIGHTS.items():
            value = item.get(field)
            if not value:
                continue
            if isinstance(value, list):
                text = " ".join(str(v) for v in value)
            else:
                text = str(value)
            tokens = _tokenize(text)
            token_count = len(tokens) or 1
            # Term frequency per field
            for token in tokens:
                token_weights[token] += (1.0 / token_count) * weight

        for token, weight in token_weights.items():
            self._inverted_index[token][item_id] = weight

        self._doc_lengths[item_id] = sum(token_weights.values()) or 1.0
        logger.debug("Indexed item %s", item_id)

    def remove_item(self, item_id: str) -> None:
        """Remove an item from the index."""
        if item_id not in self._documents:
            return
        del self._documents[item_id]
        del self._doc_lengths[item_id]
        for token_postings in self._inverted_index.values():
            token_postings.pop(item_id, None)
        logger.debug("Removed item %s from index", item_id)

    # ──────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 200,
    ) -> list[tuple[str, float]]:
        """
        Return up to *limit* (item_id, score) tuples sorted by relevance.

        *domain* filters results to items whose category belongs to the
        domain before scoring (products / sports / events).
        """
        DOMAIN_CATEGORIES: dict[str, set[str]] = {
            "products": {"electronics", "clothing", "books"},
            "sports": {"football", "basketball", "tennis"},
            "events": {"concerts", "conferences", "festivals"},
        }

        query_tokens = _tokenize(query)
        if not query_tokens:
            # No tokens – return everything optionally filtered by domain
            candidates = list(self._documents.keys())
            if domain and domain in DOMAIN_CATEGORIES:
                allowed = DOMAIN_CATEGORIES[domain]
                candidates = [
                    iid
                    for iid in candidates
                    if self._documents[iid].get("category") in allowed
                ]
            return [(iid, 0.0) for iid in candidates[:limit]]

        num_docs = len(self._documents) or 1
        scores: dict[str, float] = defaultdict(float)

        for token in query_tokens:
            postings = self._inverted_index.get(token, {})
            df = len(postings) or 1
            # IDF with smoothing
            idf = math.log((num_docs + 1) / (df + 1)) + 1.0
            for item_id, tf_weight in postings.items():
                # Normalise TF by document length
                norm_tf = tf_weight / self._doc_lengths.get(item_id, 1.0)
                scores[item_id] += norm_tf * idf

        if domain and domain in DOMAIN_CATEGORIES:
            allowed = DOMAIN_CATEGORIES[domain]
            scores = {
                iid: s
                for iid, s in scores.items()
                if self._documents.get(iid, {}).get("category") in allowed
            }

        # Also include domain-matched items with zero score so filters can act on them
        if domain and domain in DOMAIN_CATEGORIES:
            allowed = DOMAIN_CATEGORIES[domain]
            for iid, doc in self._documents.items():
                if doc.get("category") in allowed and iid not in scores:
                    scores[iid] = 0.0

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        logger.debug(
            "Query '%s' matched %d documents (domain=%s)", query, len(ranked), domain
        )
        return ranked[:limit]

    @property
    def vocabulary(self) -> list[str]:
        """All indexed tokens – used for spell correction."""
        return list(self._inverted_index.keys())

    @property
    def document_count(self) -> int:
        return len(self._documents)


# Module-level singleton
search_engine = SearchEngine()
