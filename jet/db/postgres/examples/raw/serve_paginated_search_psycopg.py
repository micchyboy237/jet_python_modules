"""
Paginated vector search with keyset pagination (FastAPI + psycopg 3 + pgvector).

Uses HNSW iterative scan for efficient nearest-neighbor search with cursor-based
pagination to avoid OFFSET performance issues.

Run:
    python serve_paginated_search_psycopg.py

Usage:
    # First page
    curl -s -X POST localhost:9001/search -H 'Content-Type: application/json' \
      -d '{"query":"machine learning artificial intelligence","limit":5,"category":"technology"}'

    # Next page (use actual cursor from response of previous page)
    curl -s -X POST localhost:9001/search -H 'Content-Type: application/json' \
      -d '{"query":"machine learning artificial intelligence","limit":5,"category":"technology","cursor":"eyJkIjowLjQ4..."}'
"""

import base64
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed, shutdown_embed_pool
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("search_api")

DB_CONNINFO = "dbname=postgres user=postgres host=localhost port=5432"
# Needs pgvector >= 0.8. Set to None to disable.
# strict_order keeps exact distance order, which keyset pagination relies on.
ITERATIVE_SCAN_MODE: Optional[str] = "strict_order"
MAX_SCAN_TUPLES = 20000


# ---------- DB pool ----------
def _configure(conn) -> None:
    """Runs once per new connection: register the vector type."""
    register_vector(conn)
    conn.commit()  # pool requires the connection to be idle after configure


pool = ConnectionPool(
    DB_CONNINFO,
    min_size=1,
    max_size=10,
    open=False,
    configure=_configure,
    kwargs={"row_factory": dict_row},
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool.open(wait=True)
    logger.info("DB pool opened")
    yield
    pool.close()
    shutdown_embed_pool()
    logger.info("DB pool and embed pool closed")


app = FastAPI(title="Paginated vector search", lifespan=lifespan)


# ---------- Schemas ----------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(10, ge=1, le=100)
    cursor: Optional[str] = None
    category: Optional[str] = None


class SearchItem(BaseModel):
    id: int
    title: str
    content: str
    category: Optional[str]
    created_at: datetime
    similarity: float


class SearchResponse(BaseModel):
    items: List[SearchItem]
    next_cursor: Optional[str]
    has_more: bool


# ---------- Cursor helpers ----------
def _fingerprint(query: str, category: Optional[str]) -> str:
    """Ties a cursor to one specific search."""
    return hashlib.sha256(f"{query}|{category}".encode()).hexdigest()[:12]


def encode_cursor(distance: float, last_id: int, fp: str) -> str:
    raw = json.dumps({"d": distance, "i": last_id, "f": fp}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode()


def decode_cursor(cursor: str, fp: str) -> Tuple[float, int]:
    try:
        data = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        distance, last_id = float(data["d"]), int(data["i"])
    except Exception:
        logger.warning("Rejected undecodable cursor")
        raise HTTPException(status_code=400, detail="Invalid cursor")
    if data.get("f") != fp:
        logger.warning("Rejected cursor from a different query")
        raise HTTPException(status_code=400, detail="Cursor does not match this query")
    return distance, last_id


# ---------- Embedding (cached so page 2+ skips the model call) ----------
@lru_cache(maxsize=256)
def _embed_query(text: str) -> tuple:
    logger.info("Embedding query (cache miss): %r", text)
    return tuple(
        embed(text=text, model=EMBED_MODEL, return_format="list", show_progress=False)
    )


# ---------- Endpoint ----------
@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    fp = _fingerprint(req.query, req.category)
    position = decode_cursor(req.cursor, fp) if req.cursor else None
    emb = list(_embed_query(req.query))

    params: dict = {"emb": emb, "limit": req.limit + 1}  # +1 to detect has_more
    conditions: List[str] = []

    if position:
        conditions.append(
            "((embedding <=> %(emb)s::vector), id) > (%(last_dist)s::float8, %(last_id)s)"
        )
        params["last_dist"], params["last_id"] = position

    if req.category:
        conditions.append("category = %(category)s")
        params["category"] = req.category

    where_sql = " AND ".join(conditions) if conditions else "TRUE"
    sql = f"""
        SELECT id, title, content, category, created_at,
               embedding <=> %(emb)s::vector AS distance
        FROM documents
        WHERE {where_sql}
        ORDER BY embedding <=> %(emb)s::vector, id
        LIMIT %(limit)s
    """

    with pool.connection() as conn:
        if ITERATIVE_SCAN_MODE:
            # SET LOCAL lasts only for this transaction
            conn.execute(f"SET LOCAL hnsw.iterative_scan = {ITERATIVE_SCAN_MODE}")
            conn.execute(f"SET LOCAL hnsw.max_scan_tuples = {int(MAX_SCAN_TUPLES)}")
        rows = conn.execute(sql, params).fetchall()

    has_more = len(rows) > req.limit
    rows = rows[: req.limit]

    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = encode_cursor(last["distance"], last["id"], fp)

    items = [
        SearchItem(
            id=r["id"],
            title=r["title"],
            content=r["content"],
            category=r["category"],
            created_at=r["created_at"],
            similarity=1.0 - r["distance"],
        )
        for r in rows
    ]
    logger.info(
        "search ok: returned=%d has_more=%s cursor_in=%s category=%s took=%.1fms",
        len(items),
        has_more,
        bool(req.cursor),
        req.category,
        (time.perf_counter() - t0) * 1000,
    )
    return SearchResponse(items=items, next_cursor=next_cursor, has_more=has_more)


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on http://0.0.0.0:9001 (docs: /docs)")
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="info")
