"""User event tracking endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from models.item import UserEvent
from services.personalization import record_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["events"])


@router.post("", status_code=201)
async def track_event(body: UserEvent) -> dict:
    """Record a user interaction (click, purchase, or skip)."""
    record_event(
        user_id=body.user_id,
        item_id=body.item_id,
        event_type=body.event_type,
    )
    logger.info(
        "Event tracked: user=%s item=%s type=%s",
        body.user_id, body.item_id, body.event_type,
    )
    return {
        "status": "recorded",
        "user_id": body.user_id,
        "item_id": body.item_id,
        "event_type": body.event_type,
    }
