"""Intent prediction API: predict what users might want next."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from services.prediction import get_predicted_interests, get_transition_probabilities

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/{user_id}")
async def predict_user_interests(
    user_id: str,
    limit: int = Query(5, ge=1, le=50),
) -> dict:
    """Predict what a user might want next based on behavior patterns.

    Uses query transition probabilities, category transitions, and
    collective intelligence to anticipate next interests.
    """
    predictions = get_predicted_interests(user_id, limit=limit)
    return {
        "user_id": user_id,
        "predictions": predictions,
        "total": len(predictions),
    }


@router.get("/transitions/query")
async def query_transitions(
    q: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=50),
) -> dict:
    """Return transition probabilities for a given query.

    Shows what users commonly search for after searching the given query,
    with probability scores.
    """
    transitions = get_transition_probabilities(q, limit=limit)
    return {
        "query": q,
        "transitions": transitions,
        "total": len(transitions),
    }
