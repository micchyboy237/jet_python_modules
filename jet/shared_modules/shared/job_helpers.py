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
    JobData,
    JobSearchResult,
    TableJobMetadata,
    TableJobRow,
)

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = EMBED_MODEL
_ctx_embd_size = get_model_ctx_embd_size(DEFAULT_EMBED_MODEL)
DEFAULT_EMBEDDING_DIM = _ctx_embd_size["embd_dims"]
# DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_JOBS_DB_NAME = "jobs_db2"
DEFAULT_TABLE_DATA = "jobs"
DEFAULT_TABLE_METADATA = "jobs_meta"
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

    # Serialize any complex values
    flat_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (dict, list)):
            flat_metadata[key] = _serialize_for_jsonb(value)
        else:
            flat_metadata[key] = value

    row_data = {"id": job_id, **flat_metadata}

    # Use create_or_update_row which handles dynamic column creation
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
            # Remove system columns
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
    Load job job for given chunk IDs or all job if no IDs provided.
    Args:
        chunk_ids: Optional list of chunk IDs to retrieve job for
        db_client: Optional PgVectorClient instance
    Returns:
        List of JobData dictionaries containing job job
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    with db_client:
        jobs: list[JobData] = db_client.get_rows(DEFAULT_TABLE_DATA, ids=chunk_ids)
    return jobs


def load_jobs_list(
    db_client: PgVectorClient | None = None,
    table_name: str = DEFAULT_TABLE_DATA,
) -> list[JobData]:
    """
    Load all existing jobs from the PostgreSQL vector database.
    Returns:
        List[JobData]: List of validated JobData objects
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    try:
        with db_client:
            rows = db_client.get_rows(
                table_name=table_name,
            )
        jobs: list[JobData] = []
        for row in rows:
            try:
                job = table_row_to_jobdata(row, db_client=db_client)
                jobs.append(job)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"Skipping invalid DB row (id={row.get('id', 'unknown')}): {e}"
                )
        logger.info(f"Loaded {len(jobs)} jobs from database table '{table_name}'")
        return jobs
    except Exception as e:
        logger.warning(f"Failed to load jobs from database: {e}")
        return []


def load_jobs_embeddings(
    chunk_ids: list[str] | None = None,
    db_client: PgVectorClient | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Reuses PgVectorClient.get_embeddings (embeddings are stored directly in the jobs table)."""
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
    Convert a database row back into a JobData object.
    Optionally joins with DEFAULT_TABLE_METADATA to load additional metadata.
    """
    chunk_meta = row.get("chunk_meta") or {}
    metadata = row.get("metadata") or {}

    job_id = metadata.get("id", row.get("id", ""))

    job_data: JobData = {
        "id": job_id,
        "link": metadata.get("link", ""),
        "title": row.get("header", ""),
        "company": metadata.get("company", row.get("parent_header", "")),
        "posted_date": metadata.get("posted_date") or row.get("posted_date"),
        "keywords": metadata.get("keywords", []),
        "details": row.get("content", ""),
        "entities": metadata.get("entities"),
        "domain": metadata.get("domain"),
        "salary": metadata.get("salary"),
        "job_type": metadata.get("job_type"),
        "hours_per_week": metadata.get("hours_per_week"),
        "tags": metadata.get("tags"),
    }

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
    Upsert one job into the vector database, optionally generating embedding if requested.
    Always returns the JobData as it exists in the database after the operation.

    Metadata (all fields except title, details, and chunk_meta) is saved to
    DEFAULT_TABLE_METADATA as a flat row.

    Args:
        job: The JobData object to save.
        db_client: (Optional) PgVectorClient for DB access.
        embed_model: (Optional) The embedding model to use.
        generate_embedding: (Optional, default False) If True, generate and store embeddings.
    """
    if db_client is None:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    ctx_embd_size = get_model_ctx_embd_size(embed_model)
    embedding_dimension = ctx_embd_size["embd_dims"]
    job_id = job["id"]
    job_hash = compute_job_hash(job)

    try:
        with db_client:
            existing = db_client.get_row(DEFAULT_TABLE_DATA, job_id)
            if existing:
                existing_chunk_meta = existing.get("chunk_meta") or {}
                if existing_chunk_meta.get("content_hash") == job_hash:
                    logger.debug(f"Job {job_id} unchanged → skipping DB update")
                    return table_row_to_jobdata(existing, db_client=db_client)
    except Exception as e:
        logger.warning(f"Error checking existing job in DB: {e}")

    text = f"{job['title'].strip()}\n{job['details'].strip()}".strip()
    embedding_array = None
    if generate_embedding:
        embedding_array = generate_embeddings([text], embed_model=embed_model)[0]

    num_tokens = count_tokens(text, model=embed_model)
    company = job.get("company", "").strip()

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

    chunk_meta_keys = set(chunk_meta.keys())

    # Job metadata for DEFAULT_TABLE_DATA (excludes title, details)
    job_metadata = {
        key: _serialize_for_jsonb(value)
        for key, value in job.items()
        if key not in ["title", "details"] and key not in chunk_meta_keys
    }

    # Flat metadata for DEFAULT_TABLE_METADATA (all fields except title, details)
    flat_metadata = {
        key: _serialize_for_jsonb(value)
        for key, value in job.items()
        if key not in ["title", "details"]
    }

    row = {
        "id": job_id,
        "header": job["title"],
        "parent_header": company,
        "content": job["details"],
        "posted_date": job.get("posted_date"),
        "chunk_meta": chunk_meta,
        "metadata": job_metadata,
        "embedding": embedding_array.tolist() if embedding_array is not None else None,
    }

    with db_client:
        saved_row = db_client.create_or_update_row(
            table_name=DEFAULT_TABLE_DATA,
            row_data=row,
            dimension=embedding_dimension,
        )

        # Save flat metadata to DEFAULT_TABLE_METADATA
        _save_metadata_to_table(db_client, job_id, flat_metadata)

        db_client.commit()
        logger.success(f"Saved/updated job {job_id} in DB (tokens: {num_tokens})")
        logger.info(
            f"Saved metadata for job {job_id} to '{DEFAULT_TABLE_METADATA}' table"
        )

        return table_row_to_jobdata(saved_row, db_client=db_client)


def save_job_embeddings(
    jobs: list[JobData],
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    overwrite_db: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME, overwrite_db=overwrite_db
        )

    ctx_embd_size = get_model_ctx_embd_size(embed_model)
    embedding_dimension = ctx_embd_size["embd_dims"]

    with db_client:
        # Ensure main data table exists
        metadata_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_DATA} (
            id              TEXT PRIMARY KEY,
            header          TEXT,
            parent_header   TEXT,
            content         TEXT,
            posted_date     TIMESTAMPTZ,
            chunk_meta      JSONB,
            metadata        JSONB,
            embedding       vector({embedding_dimension}),
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );
        """
        with db_client.conn.cursor() as cur:
            cur.execute(metadata_table_query)
            logger.debug(f"Created or verified '{DEFAULT_TABLE_DATA}' table.")

        # Ensure metadata table exists
        _ensure_metadata_table(db_client)

        existing_jobs = db_client.get_rows(DEFAULT_TABLE_DATA)
        existing_job_hashes = {}
        existing_text_hashes = {}
        for row in existing_jobs:
            chunk_meta = row.get("chunk_meta") or {}
            doc_id = chunk_meta.get("doc_id")
            if doc_id:
                existing_job_hashes[doc_id] = chunk_meta.get("content_hash")
            existing_text_hashes[row["id"]] = chunk_meta.get("text_hash")

        logger.debug(f"Existing job hashes: {len(existing_job_hashes)}")
        logger.debug(f"Existing text hashes: {len(existing_text_hashes)}")

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

    chunks_with_data = chunk_texts_with_data(
        job_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ids=[job["id"] for job, _ in jobs_to_process],
        buffer=max_job_header_token,
        model=embed_model,
    )

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

    rows_data = []
    metadata_rows = []

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

        chunk_meta_keys = set(chunk_meta.keys())
        job_metadata = {
            key: _serialize_for_jsonb(value)
            for key, value in job.items()
            if key not in ["title", "details"] and key not in chunk_meta_keys
        }

        header = job["title"]
        parent_header = job["company"]

        metadata_row = {
            "id": chunk["id"],
            "header": header,
            "parent_header": parent_header,
            "content": chunk["content"],
            "posted_date": job["posted_date"],
            "chunk_meta": chunk_meta,
            "metadata": job_metadata,
            "embedding": embedding.tolist(),
        }
        metadata_rows.append(metadata_row)

        rows_data.append(
            {
                "id": chunk["id"],
                "metadata": job_metadata,
                "text": f"{header}\n{chunk['content']}",
                "embedding": embedding,
                "content_hash": job_hash,
                "text_hash": chunk["text_hash"],
            }
        )

    with db_client:
        try:
            with db_client.conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT id FROM {} WHERE id = ANY(%s)").format(
                        sql.Identifier(DEFAULT_TABLE_DATA)
                    ),
                    ([row["id"] for row in metadata_rows],),
                )
                existing_metadata_ids = {row["id"] for row in cur.fetchall()}

            metadata_create_count = sum(
                1 for row in metadata_rows if row["id"] not in existing_metadata_ids
            )
            metadata_update_count = len(metadata_rows) - metadata_create_count

            if metadata_create_count > 0:
                logger.info(
                    f"Creating {metadata_create_count} new rows in '{DEFAULT_TABLE_DATA}' table"
                )
            if metadata_update_count > 0:
                logger.info(
                    f"Updating {metadata_update_count} existing rows in '{DEFAULT_TABLE_DATA}' table"
                )

            for idx, row in enumerate(metadata_rows):
                if "id" not in row:
                    logger.error(f"Row {idx} missing id: {row}")
                    raise ValueError(f"Row {idx} missing id")

            metadata_results = db_client.create_or_update_rows(
                DEFAULT_TABLE_DATA, metadata_rows
            )

            # Save flat metadata for each job to DEFAULT_TABLE_METADATA
            jobs_saved_metadata = set()
            for chunk, job, _ in zip(
                chunks_with_data,
                [job_by_id[chunk["doc_id"]] for chunk in chunks_with_data],
                [job_hash for _, job_hash in jobs_to_process],
            ):
                job_id = job["id"]
                if job_id not in jobs_saved_metadata:
                    flat_metadata = {
                        key: _serialize_for_jsonb(value)
                        for key, value in job.items()
                        if key not in ["title", "details"]
                    }
                    _save_metadata_to_table(db_client, job_id, flat_metadata)
                    jobs_saved_metadata.add(job_id)
                    logger.info(
                        f"Saved metadata for job {job_id} to '{DEFAULT_TABLE_METADATA}' table"
                    )

            db_client.commit()
            logger.success(
                f"Saved {len(metadata_results)} metadata records to '{DEFAULT_TABLE_DATA}' table."
            )
            logger.success(
                f"Saved metadata for {len(jobs_saved_metadata)} jobs to '{DEFAULT_TABLE_METADATA}' table."
            )
        except Exception as e:
            logger.error(f"Failed to save data: {str(e)}")
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
) -> list[JobSearchResult]:
    """
    Search for jobs based on a query string and return ranked results with data.
    Args:
        query: Search query string
        top_k: Number of top results to return
        embed_model: Embedding model to use
        db_client: Optional PgVectorClient instance
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
    return filtered_results


def hybrid_search_jobs(
    query: str,
    top_k: int | None = 10,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
) -> list[JobSearchResult]:
    from jet.vectors.reranker.bm25 import rerank_bm25

    raw_results = search_jobs(
        query=query,
        top_k=top_k,
        threshold=threshold,
        embed_model=embed_model,
        db_client=db_client,
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
    return filtered_results
