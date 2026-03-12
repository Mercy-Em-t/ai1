from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class Item(BaseModel):
    id: str
    title: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    date: Optional[str] = None  # ISO date string e.g. "2024-06-15"
    price: Optional[float] = None
    availability: bool = True
    popularity: float = 0.0   # 0-100 scale
    rating: float = 0.0       # 0-5 scale
    vector: Optional[list[float]] = None


class ItemCreate(BaseModel):
    title: str
    description: str
    category: str
    tags: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    date: Optional[str] = None
    price: Optional[float] = None
    availability: bool = True
    popularity: float = 0.0
    rating: float = 0.0
    vector: Optional[list[float]] = None


class ItemUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    location: Optional[str] = None
    date: Optional[str] = None
    price: Optional[float] = None
    availability: Optional[bool] = None
    popularity: Optional[float] = None
    rating: Optional[float] = None
    vector: Optional[list[float]] = None


class ItemResponse(BaseModel):
    id: str
    title: str
    description: str
    category: str
    tags: list[str]
    location: Optional[str]
    date: Optional[str]
    price: Optional[float]
    availability: bool
    popularity: float
    rating: float
    vector: Optional[list[float]] = None
    score: Optional[float] = None   # Search relevance score


class UserEvent(BaseModel):
    user_id: str
    item_id: str
    event_type: str = Field(..., pattern="^(click|purchase|skip)$")
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
