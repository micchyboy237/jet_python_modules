import math
from datetime import datetime

import numpy as np
from jet.adapters.llama_cpp.chunking_utils import chunk_texts_with_data
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
    TableJobRow,
    VectorSearchResult,
)

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = EMBED_MODEL
_ctx_embd_size = get_model_ctx_embd_size(DEFAULT_EMBED_MODEL)
DEFAULT_EMBEDDING_DIM = _ctx_embd_size["embd_dims"]
DEFAULT_JOBS_DB_NAME = "jobs_db3"
DEFAULT_TABLE_DATA = "jobs"  # Only stores chunked embeddings data
DEFAULT_TABLE_METADATA = "jobs_meta"  # Primary source for JobData items
DEFAULT_TABLE_ENTITIES = "job_entities"
DEFAULT_BUFFER = 32
DEFAULT_CHUNK_SIZE = _ctx_embd_size["ctx"] - DEFAULT_BUFFER
DEFAULT_CHUNK_OVERLAP = 100


def _serialize_for_jsonb(value):
    """Convert Pydantic models and other non-serializable objects to JSON-safe dicts."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _ensure_metadata_table(
    db_client: PgVectorClient, table_name: str = DEFAULT_TABLE_METADATA
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


def _save_metadata_to_table(
    db_client: PgVectorClient,
    job_id: str,
    metadata: dict,
    table_name: str = DEFAULT_TABLE_METADATA,
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
    table_name: str = DEFAULT_TABLE_METADATA,
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
            logger.debug(f"Loaded metadata for job {job_id} from '{table_name}' table.")
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


def load_jobs(
    chunk_ids: list[str] | None = None, db_client: PgVectorClient | None = None
) -> list[JobData]:
    """
    Load jobs from the metadata table. If chunk_ids provided, loads only those jobs.
    This now loads from DEFAULT_TABLE_METADATA as the primary source for JobData.

    Args:
        chunk_ids: Optional list of job IDs to retrieve
        db_client: Optional PgVectorClient instance
    Returns:
        List of JobData dictionaries
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        if chunk_ids:
            # Load specific jobs by their IDs from metadata table
            metadata_rows = db_client.get_rows(DEFAULT_TABLE_METADATA, ids=chunk_ids)
            jobs = [_metadata_row_to_jobdata(row) for row in metadata_rows]
        else:
            # Load all jobs from metadata table
            metadata_rows = db_client.get_rows(DEFAULT_TABLE_METADATA)
            jobs = [_metadata_row_to_jobdata(row) for row in metadata_rows]

    logger.info(f"Loaded {len(jobs)} jobs from '{DEFAULT_TABLE_METADATA}' table")
    return jobs


def _metadata_row_to_jobdata(row: dict) -> JobData:
    """
    Convert a metadata table row directly to JobData.
    Entities are no longer stored in jobs_meta; use load_job_entities() separately.
    """
    # DEBUG: Trace what row actually contains
    logger.debug(f"[DEBUG _metadata_row_to_jobdata] row type: {type(row)}")
    logger.debug(
        f"[DEBUG _metadata_row_to_jobdata] row keys: {list(row.keys()) if isinstance(row, dict) else 'NOT A DICT'}"
    )
    sample_keys = ["id", "title", "company", "entities"]
    for k in sample_keys:
        val = row.get(k, "<MISSING>") if isinstance(row, dict) else "<NOT DICT>"
        logger.debug(f"[DEBUG _metadata_row_to_jobdata] row['{k}'] = {repr(val)}")

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

    logger.debug(
        f"Reconstructed JobData from metadata for {job_data['id']}: "
        f"title='{job_data['title']}', company='{job_data['company']}'"
    )
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
    table_name: str = DEFAULT_TABLE_METADATA,
    include_entities: bool = False,
) -> list[JobData]:
    """
    Load all existing jobs from the metadata table.
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    try:
        with db_client:
            if include_entities:
                query = sql.SQL("""
                    SELECT m.*, e.entities AS _joined_entities
                    FROM {} m
                    LEFT JOIN {} e ON m.id = e.id
                """).format(
                    sql.Identifier(table_name),
                    sql.Identifier(DEFAULT_TABLE_ENTITIES),
                )
                with db_client.conn.cursor() as cur:
                    cur.execute(query)
                    raw_rows = cur.fetchall()

                # DEBUG: Inspect raw DB results before any processing
                logger.debug(f"[DEBUG load_jobs_list] raw_rows count: {len(raw_rows)}")
                if raw_rows:
                    first = raw_rows[0]
                    logger.debug(
                        f"[DEBUG load_jobs_list] first row type: {type(first)}"
                    )
                    if isinstance(first, dict):
                        logger.debug(
                            f"[DEBUG load_jobs_list] first row keys: {list(first.keys())}"
                        )
                        logger.debug(
                            f"[DEBUG load_jobs_list] first row id={repr(first.get('id'))}, title={repr(first.get('title'))}, _joined_entities={repr(first.get('_joined_entities'))}"
                        )
                    else:
                        cols = [d[0] for d in cur.description]
                        logger.debug(f"[DEBUG load_jobs_list] columns: {cols}")
                        logger.debug(
                            f"[DEBUG load_jobs_list] first row (tuple): {first[:5]}..."
                        )

                processed_rows = []
                for row in raw_rows:
                    if isinstance(row, dict):
                        d = dict(row)
                        d["entities"] = d.pop("_joined_entities", None)
                    else:
                        columns = [desc[0] for desc in cur.description]
                        d = dict(zip(columns, row))
                        d["entities"] = d.pop("_joined_entities", None)
                    processed_rows.append(d)

                # DEBUG: Verify processed rows
                if processed_rows:
                    first_p = processed_rows[0]
                    logger.debug(
                        f"[DEBUG load_jobs_list] after processing: id={repr(first_p.get('id'))}, title={repr(first_p.get('title'))}, entities={repr(first_p.get('entities'))}"
                    )
            else:
                processed_rows = db_client.get_rows(table_name=table_name)
                # DEBUG: Non-join path
                if processed_rows:
                    first = processed_rows[0]
                    logger.debug(
                        f"[DEBUG load_jobs_list NO JOIN] first row type: {type(first)}, id={repr(first.get('id') if isinstance(first, dict) else 'N/A')}"
                    )

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
    return db_client.get_embeddings(DEFAULT_TABLE_DATA, ids=chunk_ids)


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


def table_row_to_jobdata(
    row: TableJobRow, db_client: PgVectorClient | None = None
) -> JobData:
    """
    Convert a chunk row from DEFAULT_TABLE_DATA back into a JobData object.
    Loads full metadata from DEFAULT_TABLE_METADATA since chunk rows only contain
    chunk-specific data. Entities must be loaded separately via load_job_entities().
    """
    chunk_meta = row.get("chunk_meta") or {}
    job_id = chunk_meta.get("doc_id", row.get("id", ""))

    if not db_client:
        logger.warning(f"No db_client provided, returning minimal JobData for {job_id}")
        return {
            "id": job_id,
            "link": "",
            "title": row.get("header", ""),
            "company": row.get("parent_header", ""),
            "posted_date": row.get("posted_date"),
            "keywords": [],
            "details": row.get("content", ""),
            "entities": None,
            "domain": None,
            "salary": None,
            "job_type": None,
            "hours_per_week": None,
            "tags": None,
        }

    metadata = _load_metadata_from_table(db_client, job_id)

    if not metadata:
        logger.warning(
            f"No metadata found for job {job_id}, using chunk data as fallback"
        )
        return {
            "id": job_id,
            "link": "",
            "title": row.get("header", ""),
            "company": row.get("parent_header", ""),
            "posted_date": row.get("posted_date"),
            "keywords": [],
            "details": row.get("content", ""),
            "entities": None,
            "domain": None,
            "salary": None,
            "job_type": None,
            "hours_per_week": None,
            "tags": None,
        }

    job_data: JobData = {
        "id": job_id,
        "link": metadata.get("link", ""),
        "title": metadata.get("title", row.get("header", "")),
        "company": metadata.get("company", row.get("parent_header", "")),
        "posted_date": metadata.get("posted_date") or row.get("posted_date"),
        "keywords": metadata.get("keywords", []),
        "details": metadata.get("details", row.get("content", "")),
        "entities": None,
        "domain": metadata.get("domain"),
        "salary": metadata.get("salary"),
        "job_type": metadata.get("job_type"),
        "hours_per_week": metadata.get("hours_per_week"),
        "tags": metadata.get("tags"),
    }

    logger.debug(
        f"Reconstructed JobData for {job_id}: title='{job_data['title']}', company='{job_data['company']}'"
    )
    return job_data


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
    Save a job's metadata to DEFAULT_TABLE_METADATA only.
    Entities are excluded — use save_job_entities() separately.
    If generate_embedding is True, creates a single chunk in DEFAULT_TABLE_DATA.
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
                table_name=DEFAULT_TABLE_DATA,
                row_data=chunk_row,
                dimension=embedding_dimension,
            )
            logger.success(f"Generated embedding for job {job_id}")

        db_client.commit()
        logger.success(f"Saved/updated job {job_id} in metadata table")
        logger.info(
            f"Saved metadata for job {job_id} to '{DEFAULT_TABLE_METADATA}' table"
        )

    return _metadata_row_to_jobdata({"id": job_id, **flat_metadata})


def save_job_embeddings(
    jobs: list[JobData],
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    overwrite_db: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """
    Save chunked embeddings to DEFAULT_TABLE_DATA and full metadata to DEFAULT_TABLE_METADATA.

    DEFAULT_TABLE_DATA: Only stores chunked embeddings data
    DEFAULT_TABLE_METADATA: Stores complete JobData (all fields including title and details)
    """
    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME, overwrite_db=overwrite_db
        )
    ctx_embd_size = get_model_ctx_embd_size(embed_model)
    embedding_dimension = ctx_embd_size["embd_dims"]

    with db_client:
        # Create chunked data table (no metadata column)
        chunk_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_DATA} (
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
        with db_client.conn.cursor() as cur:
            cur.execute(chunk_table_query)
            logger.debug(f"Created or verified '{DEFAULT_TABLE_DATA}' table.")

        # Ensure metadata table exists
        _ensure_metadata_table(db_client)

        # Get existing chunk hashes for comparison
        existing_chunks = db_client.get_rows(DEFAULT_TABLE_DATA)
        existing_job_hashes = {}
        existing_text_hashes = {}
        for row in existing_chunks:
            chunk_meta = row.get("chunk_meta") or {}
            doc_id = chunk_meta.get("doc_id")
            if doc_id:
                existing_job_hashes[doc_id] = chunk_meta.get("content_hash")
            existing_text_hashes[row["id"]] = chunk_meta.get("text_hash")
        logger.debug(f"Existing job hashes: {len(existing_job_hashes)}")
        logger.debug(f"Existing text hashes: {len(existing_text_hashes)}")

    # Determine which jobs need processing
    jobs_to_process: list[tuple[JobData, str]] = []
    for job in jobs:
        job_hash = compute_job_hash(job)
        existing_hash = existing_job_hashes.get(job["id"])
        if existing_hash is None or existing_hash != job_hash:
            jobs_to_process.append((job, job_hash))
        else:
            logger.debug(f"Skipping job {job['id']} - no changes detected.")

    jobs_to_process.sort(
        key=lambda x: datetime.fromisoformat(x[0]["posted_date"]), reverse=True
    )

    if not jobs_to_process:
        logger.info("No new or changed jobs to process.")
        return {
            "chunks_with_data": [],
            "rows": [],
            "embedding_texts": [],
            "embeddings": np.array([]),
            "max_header_token": 0,
            "summary": {
                "count": 0,
                "min_token": 0,
                "ave_token": 0,
                "max_token": 0,
            },
        }

    # Save metadata for all jobs first (primary data store)
    with db_client:
        jobs_saved_metadata = set()
        for job, job_hash in jobs_to_process:
            job_id = job["id"]
            if job_id not in jobs_saved_metadata:
                flat_metadata = {
                    key: _serialize_for_jsonb(value)
                    for key, value in job.items()
                    if key != "entities"
                }
                _save_metadata_to_table(db_client, job_id, flat_metadata)
                jobs_saved_metadata.add(job_id)
                logger.info(
                    f"Saved metadata for job {job_id} to '{DEFAULT_TABLE_METADATA}' table"
                )
        db_client.commit()
        logger.success(
            f"Saved metadata for {len(jobs_saved_metadata)} jobs to '{DEFAULT_TABLE_METADATA}' table."
        )

    # Prepare text for chunking
    job_headers = []
    job_texts = []
    job_by_id = {}
    for job, job_hash in jobs_to_process:
        job_by_id[job["id"]] = job
        header = f"{job['title']}"
        job_headers.append(header)
        text = ""
        text += f"Details\n{job['details']}\n\n"
        text += f"Company: {job['company']}\n"
        if job.get("keywords"):
            text += f"Keywords: {', '.join(job['keywords'])}\n"
        if job.get("job_type"):
            text += f"Job Type: {job['job_type']}\n"
        if job.get("salary"):
            text += f"Salary: {job['salary']}\n"
        if job.get("hours_per_week"):
            text += f"Hours per Week: {job['hours_per_week']}\n"
        job_texts.append(text)

    job_header_token_counts: list[int] = count_tokens(
        job_headers, model=embed_model, prevent_total=True
    )
    max_job_header_token = (
        max(job_header_token_counts) if job_header_token_counts else 0
    )

    # Chunk the texts
    chunks_with_data = chunk_texts_with_data(
        job_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ids=[job["id"] for job, _ in jobs_to_process],
        buffer=max_job_header_token,
        model=embed_model,
    )

    # Generate chunk IDs
    for chunk in chunks_with_data:
        chunk["id"] = generate_key(
            chunk["doc_id"], chunk["chunk_index"], chunk["doc_index"]
        )
        logger.debug(
            f"Generated chunk ID: {chunk['id']} for doc_id: {chunk['doc_id']}, "
            f"chunk_index: {chunk['chunk_index']}, doc_index: {chunk['doc_index']}"
        )

    all_num_tokens = [chunk["num_tokens"] for chunk in chunks_with_data]
    count = len(chunks_with_data)
    min_token = min(all_num_tokens) if all_num_tokens else 0
    ave_token = (
        math.ceil(sum(all_num_tokens) / len(all_num_tokens)) if all_num_tokens else 0
    )
    max_token = max(all_num_tokens) if all_num_tokens else 0
    logger.log("count:", count, colors=["GRAY", "INFO"])
    logger.log("min_token:", min_token, colors=["GRAY", "SUCCESS"])
    logger.log("ave_token:", ave_token, colors=["GRAY", "SUCCESS"])
    logger.log("max_token:", max_token, colors=["GRAY", "SUCCESS"])

    # Determine which chunks need new embeddings
    chunks_to_embed = []
    embedding_texts = []
    existing_embeddings = {}
    for chunk in chunks_with_data:
        job = job_by_id.get(chunk["doc_id"])
        if not job:
            logger.error(f"No job found for doc_id: {chunk['doc_id']}")
            raise ValueError(f"No job found for doc_id: {chunk['doc_id']}")

        header = f"{job['title']}"
        text = f"{header}\n{chunk['content']}"
        text_hash = compute_text_hash(text)
        chunk["text_hash"] = text_hash

        existing_text_hash = existing_text_hashes.get(chunk["id"])
        logger.debug(
            f"Chunk ID: {chunk['id']}, Computed text_hash: {text_hash}, "
            f"Existing text_hash: {existing_text_hash}"
        )

        if existing_text_hash is None or existing_text_hash != text_hash:
            chunks_to_embed.append(chunk)
            embedding_texts.append(text)
            logger.info(
                f"Generating new embedding for chunk {chunk['id']} (new or changed content)"
            )
        else:
            with db_client:
                embedding = db_client.get_embedding_by_id(
                    DEFAULT_TABLE_DATA, chunk["id"]
                )
                if embedding is not None:
                    existing_embeddings[chunk["id"]] = embedding
                    logger.info(
                        f"Reusing existing embedding for chunk {chunk['id']} from database"
                    )
                else:
                    logger.warning(
                        f"No embedding found for unchanged chunk {chunk['id']}, will regenerate"
                    )
                    chunks_to_embed.append(chunk)
                    embedding_texts.append(text)

    # Generate new embeddings
    new_embeddings = (
        generate_embeddings(embedding_texts, embed_model)
        if embedding_texts
        else np.array([])
    )

    if len(chunks_to_embed) != len(new_embeddings):
        raise ValueError(
            f"Mismatch between chunks_to_embed ({len(chunks_to_embed)}) "
            f"and new_embeddings ({len(new_embeddings)})"
        )

    # Map embeddings to chunks
    embeddings = []
    chunk_embedding_map = {
        chunk["id"]: emb for chunk, emb in zip(chunks_to_embed, new_embeddings)
    }
    for chunk in chunks_with_data:
        if chunk["id"] in chunk_embedding_map:
            embeddings.append(chunk_embedding_map[chunk["id"]])
        elif chunk["id"] in existing_embeddings:
            embeddings.append(existing_embeddings[chunk["id"]])
        else:
            logger.error(f"No embedding found for chunk {chunk['id']}")
            raise ValueError(f"No embedding found for chunk {chunk['id']}")

    embeddings = np.array(embeddings)

    # Prepare chunk rows (no metadata field)
    rows_data = []
    chunk_rows = []
    for chunk, embedding in zip(chunks_with_data, embeddings):
        job, job_hash = next(
            (j, h) for j, h in jobs_to_process if j["id"] == chunk["doc_id"]
        )

        chunk_meta = {
            "doc_id": chunk["doc_id"],
            "header_doc_id": generate_key(job["title"]),
            "parent_id": generate_key(job["company"]),
            "doc_index": chunk["doc_index"],
            "chunk_index": chunk["chunk_index"],
            "num_tokens": chunk["num_tokens"],
            "level": 1,
            "parent_level": 0,
            "start_idx": chunk["start_idx"],
            "end_idx": chunk["end_idx"],
            "content_hash": job_hash,
            "text_hash": chunk["text_hash"],
        }

        header = job["title"]
        parent_header = job["company"]

        # Chunk row only contains chunk-specific data, no metadata
        chunk_row = {
            "id": chunk["id"],
            "header": header,
            "parent_header": parent_header,
            "content": chunk["content"],
            "posted_date": job["posted_date"],
            "chunk_meta": chunk_meta,
            "embedding": embedding.tolist(),
        }
        chunk_rows.append(chunk_row)

        rows_data.append(
            {
                "id": chunk["id"],
                "text": f"{header}\n{chunk['content']}",
                "embedding": embedding,
                "content_hash": job_hash,
                "text_hash": chunk["text_hash"],
            }
        )

    # Save chunk rows to DEFAULT_TABLE_DATA
    with db_client:
        try:
            # Check existing chunks
            with db_client.conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT id FROM {} WHERE id = ANY(%s)").format(
                        sql.Identifier(DEFAULT_TABLE_DATA)
                    ),
                    ([row["id"] for row in chunk_rows],),
                )
                existing_chunk_ids = {row["id"] for row in cur.fetchall()}

            chunk_create_count = sum(
                1 for row in chunk_rows if row["id"] not in existing_chunk_ids
            )
            chunk_update_count = len(chunk_rows) - chunk_create_count

            if chunk_create_count > 0:
                logger.info(
                    f"Creating {chunk_create_count} new chunks in '{DEFAULT_TABLE_DATA}' table"
                )
            if chunk_update_count > 0:
                logger.info(
                    f"Updating {chunk_update_count} existing chunks in '{DEFAULT_TABLE_DATA}' table"
                )

            # Validate all rows have IDs
            for idx, row in enumerate(chunk_rows):
                if "id" not in row:
                    logger.error(f"Chunk row {idx} missing id: {row}")
                    raise ValueError(f"Chunk row {idx} missing id")

            chunk_results = db_client.create_or_update_rows(
                DEFAULT_TABLE_DATA, chunk_rows
            )

            db_client.commit()
            logger.success(
                f"Saved {len(chunk_results)} chunk records to '{DEFAULT_TABLE_DATA}' table."
            )
        except Exception as e:
            logger.error(f"Failed to save chunk data: {str(e)}")
            db_client.conn.rollback()
            raise

    return {
        "chunks_with_data": chunks_with_data,
        "rows": rows_data,
        "embedding_texts": embedding_texts,
        "embeddings": embeddings,
        "max_header_token": max_job_header_token,
        "summary": {
            "count": count,
            "min_token": min_token,
            "ave_token": ave_token,
            "max_token": max_token,
        },
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


def search_jobs(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    enrich_with_metadata: bool = True,
) -> list[VectorSearchResult]:
    """
    Search for jobs based on a query string and return ranked results with data.
    Searches against chunked embeddings in DEFAULT_TABLE_DATA and optionally
    enriches results with full metadata from DEFAULT_TABLE_METADATA.

    Args:
        query: Search query string
        top_k: Number of top results to return
        threshold: Minimum score threshold
        embed_model: Embedding model to use
        db_client: Optional PgVectorClient instance
        enrich_with_metadata: If True, enrich results with full metadata from metadata table

    Returns:
        List of JobSearchResult dictionaries containing rank, score, and job data
    """
    query_embedding = generate_embeddings([query], embed_model)[0]
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        results = db_client.search(
            table_name=DEFAULT_TABLE_DATA,
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

    filtered_results = [result for result in results if is_valid_score(result["score"])]
    removed_count = len(results) - len(filtered_results)
    if removed_count > 0:
        logger.debug(f"Filtered out {removed_count} results with invalid scores")

    # Enrich results with metadata if requested
    if enrich_with_metadata:
        enriched_results = []
        for result in filtered_results:
            chunk_meta = result.get("chunk_meta", {})
            job_id = chunk_meta.get("doc_id", result.get("id", ""))

            metadata = _load_metadata_from_table(db_client, job_id)
            entity_row = load_job_entities(job_id, db_client=db_client)

            enriched = {**result}
            if metadata:
                enriched.update(
                    {
                        "job_title": metadata.get("title", result.get("header", "")),
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

        logger.debug(f"Enriched {len(enriched_results)} search results with metadata")
        return enriched_results

    return filtered_results


def hybrid_search_jobs(
    query: str,
    top_k: int | None = 10,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    enrich_with_metadata: bool = True,
) -> list[HybridSearchResult]:
    """
    Hybrid search combining vector search with BM25 reranking.
    Optionally enriches results with full metadata.
    """
    from jet.vectors.reranker.bm25 import rerank_bm25

    # Ensure db_client is available for both search and enrichment
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    raw_results = search_jobs(
        query=query,
        top_k=top_k,
        threshold=threshold,
        embed_model=embed_model,
        db_client=db_client,
        enrich_with_metadata=False,
    )

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
            "parent_header": result["parent_header"],
            "header": result["header"],
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

    # Now db_client is guaranteed to exist when enrich_with_metadata is True
    if enrich_with_metadata and db_client:
        enriched_results = []
        for result in filtered_results:
            chunk_meta = result.get("metadata", {})
            doc_id = chunk_meta.get("doc_id", "")
            if not doc_id:
                doc_id = result.get("id", "")

            metadata = _load_metadata_from_table(db_client, doc_id)
            entity_row = load_job_entities(doc_id, db_client=db_client)

            enriched = {**result}
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
            f"Enriched {len(enriched_results)} hybrid search results with metadata"
        )
        return enriched_results

    return filtered_results
