import math
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.model_utils import get_model_ctx_embd_size
from jet.adapters.llama_cpp.token_utils import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.data.utils import generate_hash, generate_key
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger
from numpy.typing import NDArray
from psycopg import sql
from shared.data_types.job import (
    HybridSearchResult,
    JobData,
    TableJobMetadata,
    VectorSearchResult,
)
from tqdm import tqdm

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = EMBED_MODEL
DEFAULT_JOBS_DB_NAME = "jobs_db3"
DEFAULT_TABLE_CHUNKS = "job_chunks"
DEFAULT_TABLE_PARENTS = "job_parents"
DEFAULT_TABLE_DATA = "jobs"
DEFAULT_TABLE_ENTITIES = "job_entities"
DEFAULT_BUFFER = 32
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def _serialize_for_jsonb(value):
    """Convert Pydantic models and other non-serializable objects to JSON-safe dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _ensure_metadata_table(
    db_client: PgVectorClient, table_name: str = DEFAULT_TABLE_DATA
) -> None:
    """
    Ensure the metadata table exists with a flat column structure.
    Columns are created dynamically when rows are inserted, but we ensure
    the base table exists with id, created_at, and updated_at.
    """
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id              TEXT PRIMARY KEY,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );
    """).format(sql.Identifier(table_name))
    with db_client.conn.cursor() as cur:
        cur.execute(query)
        logger.debug(f"Ensured metadata table '{table_name}' exists.")


def _ensure_entities_table(
    db_client: PgVectorClient, table_name: str = DEFAULT_TABLE_ENTITIES
) -> None:
    """
    Ensure the job_entities table exists with proper schema.
    Stores extracted entities with provenance metadata.
    """
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            id              TEXT PRIMARY KEY,
            model_name      TEXT,
            temperature     NUMERIC,
            extracted_at    TIMESTAMPTZ,
            entities        JSONB,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );
    """).format(sql.Identifier(table_name))
    with db_client.conn.cursor() as cur:
        cur.execute(query)
        logger.debug(f"Ensured entities table '{table_name}' exists.")


def _save_metadata_to_table(
    db_client: PgVectorClient,
    job_id: str,
    metadata: dict,
    table_name: str = DEFAULT_TABLE_DATA,
) -> None:
    """
    Save job metadata as a flat row in the metadata table.
    Each key in metadata becomes a column.
    """
    _ensure_metadata_table(db_client, table_name)

    flat_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            flat_metadata[key] = _serialize_for_jsonb(value)
        else:
            flat_metadata[key] = value

    row_data = {"id": job_id, **flat_metadata}

    db_client.create_or_update_row(table_name, row_data)
    logger.debug(f"Saved metadata for job {job_id} to '{table_name}' table.")


def _load_metadata_from_table(
    db_client: PgVectorClient,
    job_id: str,
    table_name: str = DEFAULT_TABLE_DATA,
) -> dict:
    """
    Load job metadata from the metadata table.
    Returns empty dict if not found.
    """
    try:
        row = db_client.get_row(table_name, job_id)
        if row:
            row.pop("id", None)
            row.pop("created_at", None)
            row.pop("updated_at", None)
            # logger.debug(f"Loaded metadata for job {job_id} from '{table_name}' table.")
            return row
        else:
            logger.debug(f"No metadata found for job {job_id} in '{table_name}' table.")
            return {}
    except Exception as e:
        logger.warning(f"Failed to load metadata for job {job_id}: {e}")
        return {}


def get_jobs_db_summary(db_client: PgVectorClient | None = None):
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    db_summary = db_client.get_database_summary()
    return db_summary


def _metadata_row_to_jobdata(row: dict) -> JobData:
    """
    Convert a metadata table row directly to JobData.
    """
    job_data: JobData = {
        "id": row.get("id", ""),
        "link": row.get("link", ""),
        "title": row.get("title", ""),
        "company": row.get("company", ""),
        "posted_date": row.get("posted_date"),
        "keywords": row.get("keywords", []),
        "details": row.get("details", ""),
        "entities": None,
        "domain": row.get("domain"),
        "salary": row.get("salary"),
        "job_type": row.get("job_type"),
        "hours_per_week": row.get("hours_per_week"),
        "tags": row.get("tags"),
    }
    return job_data


def save_job_entities(
    job_id: str,
    entities: dict,
    *,
    model_name: str = "qwen3.5-uncensored:2b",
    temperature: float = 0.0,
    db_client: PgVectorClient | None = None,
) -> None:
    """
    Save or update extracted entities for a job in the dedicated job_entities table.
    Includes provenance metadata (model, temperature, extraction timestamp).
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    row_data = {
        "id": job_id,
        "model_name": model_name,
        "temperature": temperature,
        "extracted_at": datetime.now().astimezone(),
        "entities": _serialize_for_jsonb(entities),
    }

    with db_client:
        db_client.create_or_update_row(DEFAULT_TABLE_ENTITIES, row_data)
        db_client.commit()

    logger.success(
        f"Saved entities for job {job_id} (model={model_name}, temp={temperature})"
    )


def load_job_entities(
    job_id: str,
    *,
    db_client: PgVectorClient | None = None,
) -> dict | None:
    """Load entities for a single job. Returns None if not found."""
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    try:
        row = db_client.get_row(DEFAULT_TABLE_ENTITIES, job_id)
        if row:
            row.pop("id", None)
            return row
        return None
    except Exception as e:
        logger.warning(f"Failed to load entities for job {job_id}: {e}")
        return None


def load_jobs_list(
    db_client: PgVectorClient | None = None,
    table_name: str = DEFAULT_TABLE_DATA,
    include_entities: bool = False,
    where_conditions: dict[str, Any] | None = None,
    posted_after: datetime | None = None,  # NEW
    posted_before: datetime | None = None,  # NEW
) -> list[JobData]:
    """
    Load jobs with optional DB-level filtering including date ranges and null checks.

    Args:
        posted_after: Only return jobs posted on or after this datetime.
        posted_before: Only return jobs posted on or before this datetime.
        where_conditions: Dict of column filters.
            - Use {"column": "NOT_NULL"} for non-null/non-empty values.
            - Use {"column": "IS_NULL"} for null/empty values.       # NEW
            - Use {"column": value} for exact match.
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    try:
        if include_entities:
            _ensure_entities_table(db_client)
        with db_client:
            if include_entities:
                base_query = sql.SQL("""
                    SELECT m.*, e.entities AS _joined_entities
                    FROM {} m
                    LEFT JOIN {} e ON m.id = e.id
                """).format(
                    sql.Identifier(table_name),
                    sql.Identifier(DEFAULT_TABLE_ENTITIES),
                )
            else:
                base_query = sql.SQL("SELECT * FROM {}").format(
                    sql.Identifier(table_name)
                )

            params: list[Any] = []
            where_parts = []

            # Handle where_conditions with IS_NULL support
            if where_conditions:
                for col, val in where_conditions.items():
                    if val == "NOT_NULL":
                        where_parts.append(
                            sql.SQL("{} IS NOT NULL AND {} != ''").format(
                                sql.Identifier(col), sql.Identifier(col)
                            )
                        )
                    elif val == "IS_NULL":  # NEW
                        where_parts.append(
                            sql.SQL("({} IS NULL OR {} = '')").format(
                                sql.Identifier(col), sql.Identifier(col)
                            )
                        )
                    else:
                        where_parts.append(
                            sql.SQL("{} = %s").format(sql.Identifier(col))
                        )
                        params.append(val)

            # Handle date range filters                               # NEW
            if posted_after is not None:
                where_parts.append(sql.SQL("posted_date >= %s"))
                params.append(posted_after.isoformat())
            if posted_before is not None:
                where_parts.append(sql.SQL("posted_date <= %s"))
                params.append(posted_before.isoformat())

            if where_parts:
                base_query = (
                    base_query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)
                )

            logger.info(
                f"Applied DB filter: {where_conditions}, after={posted_after}, before={posted_before}"
            )

            with db_client.conn.cursor() as cur:
                cur.execute(base_query, params)
                raw_rows = cur.fetchall()

            logger.debug(f"[DEBUG load_jobs_list] raw_rows count: {len(raw_rows)}")
            if raw_rows:
                first = raw_rows[0]
                logger.debug(f"[DEBUG load_jobs_list] first row type: {type(first)}")
                if isinstance(first, dict):
                    logger.debug(
                        f"[DEBUG load_jobs_list] first row keys: {list(first.keys())}"
                    )
                else:
                    cols = [d[0] for d in cur.description]
                    logger.debug(f"[DEBUG load_jobs_list] columns: {cols}")

            processed_rows = []
            for row in raw_rows:
                if isinstance(row, dict):
                    d = dict(row)
                    if include_entities:
                        d["entities"] = d.pop("_joined_entities", None)
                else:
                    columns = [desc[0] for desc in cur.description]
                    d = dict(zip(columns, row))
                    if include_entities:
                        d["entities"] = d.pop("_joined_entities", None)
                processed_rows.append(d)

            jobs: list[JobData] = []
            for row in processed_rows:
                try:
                    job = _metadata_row_to_jobdata(row)
                    if include_entities and row.get("entities") is not None:
                        job["entities"] = row["entities"]
                    jobs.append(job)
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        f"Skipping invalid metadata row (id={row.get('id', 'unknown')}): {e}"
                    )

            logger.info(
                f"Loaded {len(jobs)} jobs from '{table_name}'"
                f"{' with entities' if include_entities else ''}"
            )
            return jobs
    except Exception as e:
        logger.warning(f"Failed to load jobs from metadata table: {e}")
        return []


def load_jobs_embeddings(
    chunk_ids: list[str] | None = None,
    db_client: PgVectorClient | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Load embeddings from the chunked data table."""
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    return db_client.get_embeddings(DEFAULT_TABLE_CHUNKS, ids=chunk_ids)


def generate_embeddings(
    texts: list[str], embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL
) -> np.ndarray:
    embedder = LlamacppEmbedding(
        model=embed_model,
        use_cache=True,
        use_dynamic_batch_sizing=True,
        verbose=True,
    )
    embeddings = embedder.get_embeddings(
        texts,
        return_format="numpy",
        show_progress=True,
    )
    return embeddings


def compute_job_hash(job: JobData) -> str:
    """Compute a hash of the job's content (excluding ID) using hash_text."""
    job_copy = {k: v for k, v in job.items() if k != "id"}
    return generate_hash(job_copy)


def compute_text_hash(text: str) -> str:
    """Compute a hash of the chunk text."""
    return generate_hash(text)


def load_job_metadata(
    job_id: str,
    db_client: PgVectorClient | None = None,
) -> TableJobMetadata:
    """
    Load metadata for a specific job from the metadata table.
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    return _load_metadata_from_table(db_client, job_id)


def save_job_to_db(
    job: JobData,
    db_client: PgVectorClient | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    generate_embedding: bool = False,
) -> JobData:
    """
    Save a job's metadata to DEFAULT_TABLE_DATA only.
    Entities are excluded — use save_job_entities() separately.
    If generate_embedding is True, creates a single chunk in DEFAULT_TABLE_CHUNKS.
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    job_id = job["id"]

    # Exclude entities from metadata save — they live in job_entities table now
    flat_metadata = {
        key: _serialize_for_jsonb(value)
        for key, value in job.items()
        if key != "entities"
    }

    with db_client:
        _save_metadata_to_table(db_client, job_id, flat_metadata)

        if generate_embedding:
            ctx_embd_size = get_model_ctx_embd_size(embed_model)
            embedding_dimension = ctx_embd_size["embd_dims"]

            text = f"{job['title'].strip()}\n{job['details'].strip()}".strip()
            embedding_array = generate_embeddings([text], embed_model=embed_model)[0]
            num_tokens = count_tokens(text, model=embed_model)
            company = job.get("company", "").strip()
            job_hash = compute_job_hash(job)

            chunk_meta = {
                "doc_id": job_id,
                "header_doc_id": generate_key(job["title"]),
                "parent_id": generate_key(company) if company else None,
                "doc_index": 0,
                "chunk_index": 0,
                "num_tokens": num_tokens,
                "level": 1,
                "parent_level": 0,
                "start_idx": 0,
                "end_idx": 0,
                "content_hash": job_hash,
                "text_hash": compute_text_hash(text),
            }

            chunk_row = {
                "id": job_id,
                "header": job["title"],
                "parent_header": company,
                "content": job["details"],
                "posted_date": job.get("posted_date"),
                "chunk_meta": chunk_meta,
                "embedding": embedding_array.tolist(),
            }

            db_client.create_or_update_row(
                table_name=DEFAULT_TABLE_CHUNKS,
                row_data=chunk_row,
                dimension=embedding_dimension,
            )
            logger.success(f"Generated embedding for job {job_id}")

        db_client.commit()
        logger.success(f"Saved/updated job {job_id} in metadata table")
        logger.info(f"Saved metadata for job {job_id} to '{DEFAULT_TABLE_DATA}' table")

    return _metadata_row_to_jobdata({"id": job_id, **flat_metadata})


def save_job_embeddings(
    jobs: list[JobData],
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    overwrite_db: bool = False,
    parent_chunk_size: int | None = None,
    child_chunk_size: int | None = None,
    chunk_overlap: int = 0,
    embedding_dimension: int | None = None,
) -> dict:
    f"""
    Save PDR chunked embeddings to DEFAULT_TABLE_CHUNKS (children only),
    parent content to {DEFAULT_TABLE_PARENTS}, and full metadata to DEFAULT_TABLE_DATA.

    PDR Architecture:
      - Parents: Full logical sections stored in {DEFAULT_TABLE_PARENTS} (NOT embedded)
      - Children: Granular sentence chunks stored in jobs (embedded for search)
      - Retrieval: Match children → resolve to unique parents → full context for LLM

    Args:
        jobs: List of JobData dicts to process.
        embed_model: Embedding model key for child embeddings.
        db_client: Optional PgVectorClient instance.
        overwrite_db: If True, drop and recreate tables.
        parent_chunk_size: Max tokens per parent. None → auto-derive from LLM context.
        child_chunk_size: Max tokens per child. None → auto-derive from parent // 8.
        chunk_overlap: Overlap between consecutive children. Recommended: 0.
        embedding_dimension: Embedding dimension for the embeddings. None → auto-derive from model.

    Returns:
        Dict with processing summary including parent/child counts.
    """
    from jet.adapters.llama_cpp.chunk_strategies import ParentDocumentChunker
    from jet.adapters.llama_cpp.config import LLM_MODEL

    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME, overwrite_db=overwrite_db
        )

    if embedding_dimension is None:
        ctx_embd_size = get_model_ctx_embd_size(embed_model)
        embedding_dimension = ctx_embd_size["embd_dims"]

    # ✅ FIX: Use LLM_MODEL for parent sizing, NOT embed_model
    # embed_model (nomic-embed:2-moe) has 512 ctx → would produce parent=128, child=64
    # LLM_MODEL (qwen3.5-uncensored:2b) has 16384 ctx → produces parent=1024, child=128
    pdr_chunker = ParentDocumentChunker(model=LLM_MODEL)
    logger.info(
        f"PDR chunker initialized: parent_size={parent_chunk_size or 'auto'}, "
        f"child_size={child_chunk_size or 'auto'}, overlap={chunk_overlap}"
    )

    with db_client:
        # ── Ensure tables exist ──────────────────────────────────────────
        chunk_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_CHUNKS} (
            id              TEXT PRIMARY KEY,
            header          TEXT,
            parent_header   TEXT,
            content         TEXT,
            posted_date     TIMESTAMPTZ,
            chunk_meta      JSONB,
            embedding       vector({embedding_dimension}),
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """
        parent_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_PARENTS} (
            id              TEXT PRIMARY KEY,
            job_id          TEXT NOT NULL,
            parent_index    INT NOT NULL,
            content         TEXT NOT NULL,
            num_tokens      INT NOT NULL,
            child_ids       JSONB NOT NULL,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_job_parents_job_id
            ON {DEFAULT_TABLE_PARENTS}(job_id);
        """
        with db_client.conn.cursor() as cur:
            cur.execute(chunk_table_query)
            cur.execute(parent_table_query)
        logger.debug(
            f"Ensured '{DEFAULT_TABLE_CHUNKS}' and '{DEFAULT_TABLE_PARENTS}' tables exist."
        )

        _ensure_metadata_table(db_client)

        # ── Load existing state for dedup ────────────────────────────────
        existing_chunks = db_client.get_rows(DEFAULT_TABLE_CHUNKS)
        existing_job_hashes: dict[str, str] = {}
        existing_text_hashes: dict[str, str] = {}
        for row in existing_chunks:
            chunk_meta = row.get("chunk_meta") or {}
            doc_id = chunk_meta.get("doc_id")
            if doc_id:
                existing_job_hashes[doc_id] = chunk_meta.get("content_hash")
            existing_text_hashes[row["id"]] = chunk_meta.get("text_hash")

        logger.debug(
            f"Existing job hashes: {len(existing_job_hashes)}, "
            f"text hashes: {len(existing_text_hashes)}"
        )

        # ── Filter to new/changed jobs ───────────────────────────────────
        jobs_to_process: list[tuple[JobData, str]] = []
        for job in jobs:
            job_hash = compute_job_hash(job)
            existing_hash = existing_job_hashes.get(job["id"])
            if existing_hash is None or existing_hash != job_hash:
                jobs_to_process.append((job, job_hash))

        jobs_to_process.sort(
            key=lambda x: datetime.fromisoformat(x[0]["posted_date"]), reverse=True
        )

        if not jobs_to_process:
            logger.info("No new or changed jobs to process.")
            return {
                "parents": [],
                "children": [],
                "embedding_texts": [],
                "embeddings": np.array([]),
                "summary": {"parent_count": 0, "child_count": 0},
            }

        # ── Save metadata for all jobs being processed ───────────────────
        jobs_saved_metadata: set[str] = set()
        for job, _ in jobs_to_process:
            job_id = job["id"]
            if job_id not in jobs_saved_metadata:
                flat_metadata = {
                    k: _serialize_for_jsonb(v)
                    for k, v in job.items()
                    if k != "entities"
                }
                _save_metadata_to_table(db_client, job_id, flat_metadata)
                jobs_saved_metadata.add(job_id)
        db_client.commit()
        logger.success(
            f"Saved metadata for {len(jobs_saved_metadata)} jobs to "
            f"'{DEFAULT_TABLE_DATA}' table."
        )

        # ── PDR Chunking ─────────────────────────────────────────────────
        all_parents: list[dict] = []
        all_children: list[dict] = []
        job_by_child_id: dict[str, tuple[JobData, str]] = {}

        for job, job_hash in jobs_to_process:
            job_id = job["id"]

            # Build composite text (same structure as before)
            text_parts = [f"Details\n{job['details']}\n"]
            text_parts.append(f"Company: {job['company']}\n")
            if job.get("keywords"):
                text_parts.append(f"Keywords: {', '.join(job['keywords'])}\n")
            if job.get("job_type"):
                text_parts.append(f"Job Type: {job['job_type']}\n")
            if job.get("salary"):
                text_parts.append(f"Salary: {job['salary']}\n")
            if job.get("hours_per_week"):
                text_parts.append(f"Hours per Week: {job['hours_per_week']}\n")
            job_text = "".join(text_parts)

            # Generate PDR parent-child pairs
            pdr_result = pdr_chunker.chunk_pdr(
                text=job_text,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                chunk_overlap=chunk_overlap,
            )

            # Tag parents/children with job_id for DB storage
            for parent in pdr_result["parents"]:
                parent["job_id"] = job_id
                all_parents.append(parent)

            for child in pdr_result["children"]:
                child["doc_id"] = job_id
                all_children.append(child)
                job_by_child_id[child["id"]] = (job, job_hash)

        logger.info(
            f"PDR chunking complete: {len(all_parents)} parents, "
            f"{len(all_children)} children from {len(jobs_to_process)} jobs"
        )

        # ── Clean up stale data for re-processed jobs ────────────────────
        reprocessed_job_ids = [j["id"] for j, _ in jobs_to_process]
        if reprocessed_job_ids:
            with db_client.conn.cursor() as cur:
                # Delete old children
                cur.execute(
                    sql.SQL(
                        "DELETE FROM {} WHERE chunk_meta->>'doc_id' = ANY(%s)"
                    ).format(sql.Identifier(DEFAULT_TABLE_CHUNKS)),
                    (reprocessed_job_ids,),
                )
                deleted_children = cur.rowcount
                # Delete old parents
                cur.execute(
                    sql.SQL(
                        f"DELETE FROM {DEFAULT_TABLE_PARENTS} WHERE job_id = ANY(%s)"
                    ),
                    (reprocessed_job_ids,),
                )
                deleted_parents = cur.rowcount
            db_client.commit()
            logger.info(
                f"Cleaned up {deleted_children} old children and "
                f"{deleted_parents} old parents for {len(reprocessed_job_ids)} jobs"
            )

        # ── Save parents to {DEFAULT_TABLE_PARENTS} ──────────────────────────────────
        parent_rows = []
        for parent in all_parents:
            parent_rows.append(
                {
                    "id": parent["id"],
                    "job_id": parent["job_id"],
                    "parent_index": parent["parent_chunk_index"],
                    "content": parent["content"],
                    "num_tokens": parent["num_tokens"],
                    "child_ids": parent["child_ids"],
                }
            )

        if parent_rows:
            db_client.create_or_update_rows(DEFAULT_TABLE_PARENTS, parent_rows)
            db_client.commit()
            logger.success(
                f"Saved {len(parent_rows)} parent records to '{DEFAULT_TABLE_PARENTS}' table"
            )

        # ── Prepare children for embedding ───────────────────────────────
        chunks_to_embed: list[dict] = []
        embedding_texts: list[str] = []
        existing_embeddings: dict[str, list[float]] = {}

        for child in all_children:
            job, _ = job_by_child_id[child["id"]]
            header = job["title"]
            embed_text = f"{header}\n{child['content']}"
            text_hash = compute_text_hash(embed_text)
            child["text_hash"] = text_hash

            existing_text_hash = existing_text_hashes.get(child["id"])
            if existing_text_hash is not None and existing_text_hash == text_hash:
                # Reuse existing embedding if content unchanged
                cached_emb = db_client.get_embedding_by_id(
                    DEFAULT_TABLE_CHUNKS, child["id"]
                )
                if cached_emb is not None:
                    existing_embeddings[child["id"]] = cached_emb
                    logger.debug(
                        f"Reusing embedding for child {child['id']} (hash match)"
                    )
                    continue
                else:
                    logger.warning(
                        f"No cached embedding for unchanged child {child['id']}, "
                        f"regenerating"
                    )

            chunks_to_embed.append(child)
            embedding_texts.append(embed_text)

        logger.info(
            f"Embedding {len(chunks_to_embed)} new/changed children "
            f"(reused {len(all_children) - len(chunks_to_embed)})"
        )

        # ── Batch embed children ─────────────────────────────────────────
        new_embeddings = (
            generate_embeddings(embedding_texts, embed_model)
            if embedding_texts
            else np.array([])
        )

        if len(chunks_to_embed) != len(new_embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks_to_embed)} chunks vs "
                f"{len(new_embeddings)} embeddings"
            )

        # Map embeddings back to all children
        new_emb_map = {c["id"]: emb for c, emb in zip(chunks_to_embed, new_embeddings)}
        child_embedding_map: dict[str, np.ndarray] = {}
        for child in all_children:
            if child["id"] in new_emb_map:
                child_embedding_map[child["id"]] = new_emb_map[child["id"]]
            elif child["id"] in existing_embeddings:
                child_embedding_map[child["id"]] = np.array(
                    existing_embeddings[child["id"]]
                )
            else:
                raise ValueError(f"No embedding found for child {child['id']}")

        # ── Save child chunks to jobs table ──────────────────────────────
        chunk_rows = []
        for child in all_children:
            job, job_hash = job_by_child_id[child["id"]]
            embedding = child_embedding_map[child["id"]]

            chunk_meta = {
                "doc_id": child["doc_id"],
                "header_doc_id": generate_key(job["title"]),
                "parent_id": child["parent_id"],
                "doc_index": 0,
                "chunk_index": child["child_index_within_parent"],
                "num_tokens": child["num_tokens"],
                "level": 1,
                "parent_level": 0,
                "start_idx": 0,
                "end_idx": 0,
                "content_hash": job_hash,
                "text_hash": child["text_hash"],
                "chunk_role": "child",
                "parent_chunk_index": child["parent_chunk_index"],
                "child_index_within_parent": child["child_index_within_parent"],
            }

            chunk_rows.append(
                {
                    "id": child["id"],
                    "header": job["title"],
                    "parent_header": job["company"],
                    "content": child["content"],
                    "posted_date": job["posted_date"],
                    "chunk_meta": chunk_meta,
                    "embedding": embedding.tolist(),
                }
            )

        if chunk_rows:
            db_client.create_or_update_rows(DEFAULT_TABLE_CHUNKS, chunk_rows)
            db_client.commit()
            logger.success(
                f"Saved {len(chunk_rows)} child chunk records to "
                f"'{DEFAULT_TABLE_CHUNKS}' table"
            )

    # ── Summary stats ────────────────────────────────────────────────────
    child_token_counts = [c["num_tokens"] for c in all_children]
    parent_token_counts = [p["num_tokens"] for p in all_parents]

    summary = {
        "parent_count": len(all_parents),
        "child_count": len(all_children),
        "jobs_processed": len(jobs_to_process),
        "parent_tokens": {
            "min": min(parent_token_counts) if parent_token_counts else 0,
            "avg": math.ceil(sum(parent_token_counts) / len(parent_token_counts))
            if parent_token_counts
            else 0,
            "max": max(parent_token_counts) if parent_token_counts else 0,
        },
        "child_tokens": {
            "min": min(child_token_counts) if child_token_counts else 0,
            "avg": math.ceil(sum(child_token_counts) / len(child_token_counts))
            if child_token_counts
            else 0,
            "max": max(child_token_counts) if child_token_counts else 0,
        },
    }

    logger.info(f"PDR embedding summary: {summary}")

    return {
        "parents": all_parents,
        "children": all_children,
        "embedding_texts": embedding_texts,
        "embeddings": new_embeddings,
        "summary": summary,
    }


def is_valid_score(score) -> bool:
    """Check if score is a valid positive number (not NaN, None, or zero)."""
    if score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    if math.isnan(score) or math.isinf(score):
        return False
    return score > 0


def _resolve_parents_from_children(
    results: list[dict],
    db_client: PgVectorClient,
) -> dict[str, str]:
    f"""
    Resolve child search results to their parent content.

    Extracts unique parent_ids from chunk_meta, batch-fetches from
    {DEFAULT_TABLE_PARENTS} table, and returns a mapping of parent_id → content.

    Args:
        results: List of search result dicts with chunk_meta containing parent_id.
        db_client: Active PgVectorClient instance.

    Returns:
        Dict mapping parent_id to full parent content string.
    """
    parent_ids: set[str] = set()
    for result in results:
        chunk_meta = result.get("chunk_meta") or result.get("metadata", {})
        pid = chunk_meta.get("parent_id")
        if pid:
            parent_ids.add(pid)

    if not parent_ids:
        logger.debug("No parent_ids found in search results")
        return {}

    try:
        parent_rows = db_client.get_rows(DEFAULT_TABLE_PARENTS, ids=list(parent_ids))
        parent_map = {row["id"]: row["content"] for row in parent_rows}
        missing = parent_ids - set(parent_map.keys())
        if missing:
            logger.warning(
                f"Missing {len(missing)} parents in {DEFAULT_TABLE_PARENTS} table: "
                f"{list(missing)[:5]}..."
            )
        logger.debug(
            f"Resolved {len(parent_map)}/{len(parent_ids)} parents from {DEFAULT_TABLE_PARENTS}"
        )
        return parent_map
    except Exception as e:
        logger.error(f"Failed to resolve parents from {DEFAULT_TABLE_PARENTS}: {e}")
        return {}


def search_jobs(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    enrich_with_metadata: bool = True,
) -> list[VectorSearchResult]:
    f"""
    Search for jobs based on a query string and return ranked results with data.
    Searches against CHILD embeddings in DEFAULT_TABLE_CHUNKS, resolves to
    PARENT content from {DEFAULT_TABLE_PARENTS} for full-context retrieval.
    """
    query_embedding = generate_embeddings([query], embed_model)[0]
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        results = db_client.search(
            table_name=DEFAULT_TABLE_CHUNKS,
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )
        filtered_results = [r for r in results if is_valid_score(r["score"])]
        removed_count = len(results) - len(filtered_results)
        if removed_count > 0:
            logger.debug(f"Filtered out {removed_count} results with invalid scores")

        if enrich_with_metadata:
            # ← NEW: Resolve parent content for all child hits
            parent_map = _resolve_parents_from_children(filtered_results, db_client)

            enriched_results = []
            # <--- Wrap loop with tqdm
            for result in tqdm(filtered_results, desc="Enriching results"):
                chunk_meta = result.get("chunk_meta", {})
                job_id = chunk_meta.get("doc_id", result.get("id", ""))
                metadata = _load_metadata_from_table(db_client, job_id)
                entity_row = load_job_entities(job_id, db_client=db_client)

                # ← NEW: Attach parent content (falls back to child content)
                parent_id = chunk_meta.get("parent_id")
                parent_content = parent_map.get(parent_id, result.get("content", ""))

                enriched = {**result}
                enriched["parent_content"] = parent_content  # ← NEW

                if metadata:
                    enriched.update(
                        {
                            "job_title": metadata.get(
                                "title", result.get("header", "")
                            ),
                            "company": metadata.get(
                                "company", result.get("parent_header", "")
                            ),
                            "link": metadata.get("link", ""),
                            "keywords": metadata.get("keywords", []),
                            "entities": entity_row["entities"] if entity_row else None,
                            "domain": metadata.get("domain"),
                            "salary": metadata.get("salary"),
                            "job_type": metadata.get("job_type"),
                            "tags": metadata.get("tags"),
                            "hours_per_week": metadata.get("hours_per_week"),
                        }
                    )
                enriched_results.append(enriched)

            # logger.debug( # <--- Optional: Remove or keep this summary log
            #     f"Enriched {len(enriched_results)} search results with metadata + parent content"
            # )
            return enriched_results

        return filtered_results


# Job-Level Result Type
class JobSearchResult(TypedDict, total=False):
    """Job-level search result derived from public.jobs + public.job_entities."""

    id: str
    rank: int
    score: float
    distance: float
    link: str
    title: str
    company: str
    posted_date: str | None
    keywords: list[str] | None
    details: str | None
    domain: str | None
    salary: str | None
    job_type: str | None
    hours_per_week: str | None
    tags: list[str] | None
    entities: dict | None


def search_full_jobs(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    posted_after: datetime | None = None,  # NEW
    posted_before: datetime | None = None,  # NEW
    where_conditions: dict[str, Any] | None = None,  # NEW
) -> list[JobSearchResult]:
    """
    Search for jobs and return unique Job-Level results.
    Now supports date range and null/non-null filtering at the DB level.
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    # 1. Use existing search_jobs to get chunk-level matches
    # Fetch extra to account for deduplication reducing count
    fetch_limit = (top_k * 3) if top_k else 100

    chunk_results = search_jobs(
        query=query,
        top_k=fetch_limit,
        threshold=threshold,
        embed_model=embed_model,
        db_client=db_client,
        enrich_with_metadata=False,
    )

    if not chunk_results:
        return []

    # 2. Deduplicate by doc_id, keeping highest score per job
    best_scores: dict[str, float] = {}
    for result in chunk_results:
        chunk_meta = result.get("chunk_meta", {})
        job_id = chunk_meta.get("doc_id")
        if not job_id:
            continue
        score = result.get("score", 0.0)
        if job_id not in best_scores or score > best_scores[job_id]:
            best_scores[job_id] = score

    if not best_scores:
        return []

    # 3. Sort by score and apply top_k limit
    sorted_job_ids = sorted(
        best_scores.keys(), key=lambda jid: best_scores[jid], reverse=True
    )
    if top_k:
        sorted_job_ids = sorted_job_ids[:top_k]

    # 4. Batch load jobs WITH entities from DB
    # This ensures we get the canonical job record, not chunk artifacts
    jobs_with_entities = load_jobs_list(
        db_client=db_client,
        include_entities=True,
        posted_after=posted_after,
        posted_before=posted_before,
        where_conditions=where_conditions,
    )

    jobs_map = {job["id"]: job for job in jobs_with_entities}

    # 5. Construct Job-Level Results
    final_results: list[JobSearchResult] = []
    for rank, job_id in enumerate(sorted_job_ids, start=1):
        job = jobs_map.get(job_id)
        if not job:
            logger.warning(
                f"Job {job_id} found in vector index but missing in public.jobs"
            )
            continue

        score = best_scores[job_id]
        result: JobSearchResult = {
            "id": job_id,
            "rank": rank,
            "score": score,
            "distance": round(1.0 - score, 6),
            "link": job.get("link"),
            "title": job.get("title"),
            "company": job.get("company"),
            "posted_date": job.get("posted_date"),
            "keywords": job.get("keywords"),
            "details": job.get("details"),
            "domain": job.get("domain"),
            "salary": job.get("salary"),
            "job_type": job.get("job_type"),
            "hours_per_week": job.get("hours_per_week"),
            "tags": job.get("tags"),
            "entities": job.get("entities"),
        }
        final_results.append(result)

    logger.info(
        f"Deduplicated search: {len(final_results)} unique jobs from {len(chunk_results)} chunks"
    )
    return final_results


def filter_jobs_by_metadata(
    where_conditions: dict[str, Any] | None = None,
    title_ilike: str | None = None,
    details_ilike: str | None = None,
    limit: int | None = None,
    db_client: PgVectorClient | None = None,
) -> list[str]:
    """
    Filter jobs at the DB level using metadata columns and optional text search.
    Returns list of matching job IDs.

    Args:
        where_conditions: Exact-match column filters (e.g., {"job_type": "Full-time"})
        title_ilike: Case-insensitive LIKE pattern for title column (e.g., "%junior%")
        details_ilike: Case-insensitive LIKE pattern for details column
        limit: Max number of job IDs to return
        db_client: Optional PgVectorClient instance

    Returns:
        List of job IDs matching the filters
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    # Build combined where conditions
    combined_where = dict(where_conditions) if where_conditions else {}

    # For ILIKE filters, we need raw SQL since get_rows only supports exact match
    has_text_filter = title_ilike or details_ilike

    if has_text_filter:
        ilike_clauses = []
        params: list[Any] = []
        if title_ilike:
            ilike_clauses.append("title ILIKE %s")
            params.append(title_ilike)
        if details_ilike:
            ilike_clauses.append("details ILIKE %s")
            params.append(details_ilike)

        where_parts = []
        for col, val in combined_where.items():
            where_parts.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
            params.append(val)

        if ilike_clauses:
            where_parts.append(
                sql.SQL("(")
                + sql.SQL(" OR ").join(map(sql.SQL, ilike_clauses))
                + sql.SQL(")")
            )

        full_where = (
            sql.SQL(" AND ").join(where_parts) if where_parts else sql.SQL("TRUE")
        )

        query = sql.SQL("SELECT id FROM {} WHERE {}").format(
            sql.Identifier(DEFAULT_TABLE_DATA),
            full_where,
        )
        if limit:
            query = sql.SQL("{} LIMIT %s").format(query)
            params.append(limit)

        with db_client.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        job_ids = [row["id"] for row in rows]
        logger.info(
            f"DB metadata filter: {len(job_ids)} jobs matched "
            f"(where={combined_where}, title_ilike={title_ilike}, details_ilike={details_ilike})"
        )
        return job_ids
    else:
        # Use existing get_rows for exact-match-only filters
        rows = db_client.get_rows(
            DEFAULT_TABLE_DATA,
            where_conditions=combined_where if combined_where else None,
            limit=limit,
        )
        job_ids = [row["id"] for row in rows]
        logger.info(
            f"DB metadata filter: {len(job_ids)} jobs matched (where={combined_where})"
        )
        return job_ids


def hybrid_search_jobs(
    query: str,
    top_k: int | None = 10,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    enrich_with_metadata: bool = True,
    metadata_filters: dict[str, Any] | None = None,
    title_ilike: str | None = None,
    details_ilike: str | None = None,
) -> list[HybridSearchResult]:
    """
    Hybrid search combining vector search with BM25 reranking.
    Optionally pre-filters candidates at the DB metadata level before vector search.

    NEW ARGS:
        metadata_filters: Dict of exact-match column filters applied via SQL WHERE
                          before vector search (e.g., {"job_type": "Full-time"})
        title_ilike: Case-insensitive LIKE pattern for title pre-filtering
        details_ilike: Case-insensitive LIKE pattern for details pre-filtering
    """
    from jet.vectors.reranker.bm25 import rerank_bm25

    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    # --- NEW: Pre-filter at DB level if filters provided ---
    candidate_ids: list[str] | None = None
    if metadata_filters or title_ilike or details_ilike:
        candidate_ids = filter_jobs_by_metadata(
            where_conditions=metadata_filters,
            title_ilike=title_ilike,
            details_ilike=details_ilike,
            limit=top_k * 5 if top_k else None,
            db_client=db_client,
        )
        if not candidate_ids:
            logger.warning(
                "DB pre-filter returned 0 candidates; returning empty results"
            )
            return []
        logger.info(
            f"Pre-filtered to {len(candidate_ids)} candidate job IDs for hybrid search"
        )

    # If we have candidate IDs, search only those; otherwise standard search
    if candidate_ids is not None:
        # Load embeddings only for filtered candidates
        candidate_embeddings = db_client.get_embeddings(
            DEFAULT_TABLE_CHUNKS, ids=candidate_ids
        )
        if not candidate_embeddings:
            logger.warning("No embeddings found for pre-filtered candidate IDs")
            return []

        # Build documents/metadata from candidate chunks
        raw_results = []
        for chunk_id in candidate_ids:
            emb = candidate_embeddings.get(chunk_id)
            if emb is None:
                continue
            row = db_client.get_row(DEFAULT_TABLE_CHUNKS, chunk_id)
            if not row:
                continue
            raw_results.append(
                {
                    "id": chunk_id,
                    "content": row.get("content", ""),
                    "header": row.get("header", ""),
                    "parent_header": row.get("parent_header", ""),
                    "chunk_meta": row.get("chunk_meta", {}),
                    "score": 1.0,  # placeholder; rerank will re-score
                }
            )
    else:
        raw_results = search_jobs(
            query=query,
            top_k=top_k,
            threshold=threshold,
            embed_model=embed_model,
            db_client=db_client,
            enrich_with_metadata=False,
        )

    # --- Existing BM25 rerank logic (unchanged) ---
    ids = [result["id"] for result in raw_results]
    documents = [f"{result['content']}" for result in raw_results]
    metadatas = [
        {
            "parent_id": result["chunk_meta"].get("parent_id"),
            "doc_id": result["chunk_meta"].get("doc_id"),
            "chunk_index": result["chunk_meta"].get("chunk_index"),
            "start_idx": result["chunk_meta"].get("start_idx"),
            "end_idx": result["chunk_meta"].get("end_idx"),
            "num_tokens": result["chunk_meta"].get("num_tokens"),
            "header": result.get("header", ""),
            "parent_header": result.get("parent_header", ""),
        }
        for result in raw_results
    ]

    query_candidates, reranked_results = rerank_bm25(query, documents, ids, metadatas)

    filtered_results = [
        result for result in reranked_results if is_valid_score(result.get("score"))
    ]
    removed_count = len(reranked_results) - len(filtered_results)
    if removed_count > 0:
        logger.debug(
            f"Filtered out {removed_count} reranked results with invalid scores"
        )

    if enrich_with_metadata and db_client:
        # ← NEW: Resolve parent content for all child hits
        parent_map = _resolve_parents_from_children(filtered_results, db_client)

        enriched_results = []
        for result in filtered_results:
            chunk_meta = result.get("metadata", {})
            doc_id = chunk_meta.get("doc_id", "")
            if not doc_id:
                doc_id = result.get("id", "")
            metadata = _load_metadata_from_table(db_client, doc_id)
            entity_row = load_job_entities(doc_id, db_client=db_client)

            # ← Attach parent content
            parent_id = chunk_meta.get("parent_id")
            parent_content = parent_map.get(parent_id, result.get("text", ""))

            enriched = {**result}
            enriched["parent_content"] = parent_content

            # Extract header/parent_header from metadata to top-level
            enriched["header"] = chunk_meta.pop("header", "")
            enriched["parent_header"] = chunk_meta.pop("parent_header", "")

            enriched["metadata"] = chunk_meta

            if metadata:
                enriched.update(
                    {
                        "job_title": metadata.get("title", ""),
                        "company": metadata.get("company", ""),
                        "link": metadata.get("link", ""),
                        "keywords": metadata.get("keywords", []),
                        "entities": entity_row["entities"] if entity_row else None,
                        "domain": metadata.get("domain"),
                        "salary": metadata.get("salary"),
                        "job_type": metadata.get("job_type"),
                        "tags": metadata.get("tags"),
                        "hours_per_week": metadata.get("hours_per_week"),
                    }
                )
            else:
                logger.warning(
                    f"No metadata found for doc_id='{doc_id}' (chunk_id={result.get('id')})"
                )
            enriched_results.append(enriched)

        logger.debug(
            f"Enriched {len(enriched_results)} hybrid search results with metadata + parent content"
        )
        return enriched_results

    return filtered_results
