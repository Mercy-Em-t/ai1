# Hybrid Search Engine

A FastAPI-based hybrid search engine combining TF-IDF full-text search, intent detection, synonym expansion, personalisation, and multi-factor ranking — with Redis caching (in-memory fallback) and no external search service required.

---

## Quick Start

```bash
cd hybrid-search-engine
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## API Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |

### Items (CRUD)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/items` | Create a new item |
| GET | `/items` | List items (optional `?category=`) |
| GET | `/items/{item_id}` | Retrieve a single item |
| PUT | `/items/{item_id}` | Update an item |
| DELETE | `/items/{item_id}` | Delete an item |

### Search
| Method | Path | Description |
|--------|------|-------------|
| GET | `/search` | Full hybrid search pipeline |

**Search query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | required | Search query |
| `location` | string | — | Filter by location (substring) |
| `min_price` | float | — | Minimum price |
| `max_price` | float | — | Maximum price |
| `category` | string | — | Exact category match |
| `availability` | bool | — | `true` / `false` |
| `min_date` | string | — | ISO date lower bound |
| `max_date` | string | — | ISO date upper bound |
| `page` | int | 1 | Page number |
| `page_size` | int | 10 | Results per page (max 100) |
| `user_id` | string | — | Enable personalised ranking |

**Example:**
```
GET /search?q=tennis+racket&max_price=300&availability=true&user_id=user123
```

**Response:**
```json
{
  "results": [...],
  "total": 4,
  "page": 1,
  "page_size": 10,
  "corrected_query": null,
  "domain": "sports",
  "query_time_ms": 3.14
}
```

### Events
| Method | Path | Description |
|--------|------|-------------|
| POST | `/events` | Record a user interaction |

**Body:**
```json
{
  "user_id": "user123",
  "item_id": "sport-013",
  "event_type": "click"
}
```
`event_type` must be one of: `click`, `purchase`, `skip`.

---

## Architecture

```
Request
  │
  ├─ Phase 1: Spell Correction      (utils/spell.py)
  ├─ Phase 2: Intent Detection      (services/intent.py)
  ├─ Phase 3: Synonym Expansion     (utils/synonyms.py)
  ├─ Phase 4: Candidate Generation  (services/candidates.py + search/engine.py)
  ├─ Phase 5: Filtering             (filters/filter.py)
  ├─ Phase 6: Ranking               (ranking/ranker.py)
  ├─ Phase 7: Personalisation       (ranking/ranker.py + services/personalization.py)
  ├─ Phase 8: Pagination
  └─ Phase 9: Response + Cache      (services/cache.py)
```

### Key Components

| Module | Responsibility |
|--------|---------------|
| `search/engine.py` | In-memory TF-IDF inverted index |
| `ranking/ranker.py` | Composite scoring (relevance 40%, popularity 30%, rating 20%, recency 10%) |
| `filters/filter.py` | Location, date, price, availability, category filters |
| `services/cache.py` | Redis with in-memory fallback (TTL 300 s) |
| `services/intent.py` | Keyword-based query classification |
| `utils/synonyms.py` | Domain-specific synonym map |
| `utils/spell.py` | `difflib`-based fuzzy spell correction |
| `data/sample_data.py` | 50 sample items (products, sports, events) |

### Data Model

```python
Item:
  id: str
  title: str
  description: str
  category: str
  tags: list[str]
  location: str | None
  date: str | None        # ISO 8601
  price: float | None
  availability: bool
  popularity: float       # 0–100
  rating: float           # 0–5
  vector: list[float] | None  # reserved for semantic search
```

### Caching

Set the `REDIS_URL` environment variable to point to your Redis instance:

```bash
export REDIS_URL=redis://localhost:6379
uvicorn main:app --reload
```

If Redis is unreachable the engine transparently falls back to an in-process dict cache with the same TTL semantics.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
