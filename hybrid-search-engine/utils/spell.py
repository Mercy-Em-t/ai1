"""Spell correction using difflib fuzzy matching."""
from __future__ import annotations

import difflib
import logging
import re

logger = logging.getLogger(__name__)


def _tokenize_query(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def correct_query(query: str, vocabulary: list[str]) -> tuple[str, bool]:
    """
    Attempt to correct misspelled tokens in *query* using *vocabulary*.

    Returns a ``(corrected_query, was_corrected)`` tuple.
    If no corrections are made, returns the original query unchanged.
    """
    if not vocabulary:
        return query, False

    vocab_set = set(vocabulary)
    tokens = _tokenize_query(query)
    corrected_tokens: list[str] = []
    any_corrected = False

    for token in tokens:
        if token in vocab_set or len(token) <= 2:
            corrected_tokens.append(token)
            continue

        matches = difflib.get_close_matches(token, vocabulary, n=1, cutoff=0.8)
        if matches and matches[0] != token:
            logger.debug("Spell correction: '%s' → '%s'", token, matches[0])
            corrected_tokens.append(matches[0])
            any_corrected = True
        else:
            corrected_tokens.append(token)

    corrected = " ".join(corrected_tokens)
    return corrected, any_corrected
