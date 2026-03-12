"""Redis caching with graceful in-memory fallback."""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# In-memory fallback cache: key → (value, expiry_timestamp)
_memory_cache: dict[str, tuple[Any, float]] = {}

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis

        client = redis.from_url(_REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis at %s", _REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s), using in-memory cache", exc)
        _redis_client = None
    return _redis_client


def get_cache(key: str) -> Any | None:
    """Return cached value or ``None`` if missing / expired."""
    client = _get_redis()
    if client is not None:
        try:
            raw = client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis get error: %s", exc)

    # Fallback
    entry = _memory_cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if expiry and time.time() > expiry:
        del _memory_cache[key]
        return None
    return value


def set_cache(key: str, value: Any, ttl: int = 300) -> None:
    """Store *value* under *key* with a TTL in seconds."""
    client = _get_redis()
    if client is not None:
        try:
            client.setex(key, ttl, json.dumps(value))
            return
        except Exception as exc:
            logger.warning("Redis set error: %s", exc)

    # Fallback
    expiry = time.time() + ttl if ttl else 0.0
    _memory_cache[key] = (value, expiry)


def invalidate_cache(key: str) -> None:
    """Delete *key* from the cache."""
    client = _get_redis()
    if client is not None:
        try:
            client.delete(key)
            return
        except Exception as exc:
            logger.warning("Redis delete error: %s", exc)

    _memory_cache.pop(key, None)
