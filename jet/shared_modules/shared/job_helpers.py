import math
from datetime import datetime
from typing import cast

import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_EMBEDDING_SIZES
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.data.utils import generate_hash, generate_key
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger
from jet.models.tokenizer.base import count_tokens
from jet.wordnet.text_chunker import chunk_texts_with_data
from numpy.typing import NDArray
from psycopg import sql
from shared.data_types.job import JobData, JobSearchResult

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = "nomic-embed-text-v2-moe"
DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_TABLE_DATA_NAME = f"{DEFAULT_TABLE_NAME}_data"
DEFAULT_JOBS_TABLE = "jobs"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100

# Keys stored in the jobs table (exclude created_at/updated_at when building JobData)
JOBS_TABLE_DATA_KEYS = (
    "id",
    "link",
    "title",
    "company",
    "posted_date",
    "details",
    "keywords",
    "entities",
    "domain",
    "salary",
    "job_type",
    "hours_per_week",
    "tags",
    "meta",
)


def get_jobs_db_summary(db_client: PgVectorClient | None = None):
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    db_summary = db_client.get_database_summary()
    return db_summary


def _ensure_jobs_table_exists(db_client: PgVectorClient) -> None:
    """Create the jobs table if it does not exist (no embedding column)."""
    with db_client.conn.cursor() as cur:
        cur.execute(
            sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                link TEXT,
                title TEXT,
                company TEXT,
                posted_date TEXT,
                details TEXT,
                keywords JSONB,
                entities JSONB,
                domain TEXT,
                salary TEXT,
                job_type TEXT,
                hours_per_week INTEGER,
                tags JSONB,
                meta JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """).format(sql.Identifier(DEFAULT_JOBS_TABLE))
        )
    logger.debug(f"Ensured table '{DEFAULT_JOBS_TABLE}' exists.")


def _job_to_row(job: dict) -> dict:
    """Convert a job dict (or dataclass __dict__) to a DB row with only allowed keys."""
    row = {"id": job["id"]}
    for key in JOBS_TABLE_DATA_KEYS:
        if key == "id":
            continue
        if key in job and job[key] is not None:
            row[key] = job[key]
        else:
            # Ensure all columns present for create_or_update_rows consistency
            if key in ("keywords", "entities", "tags", "meta"):
                row[key] = job.get(key) or {}
            elif key == "details":
                row[key] = job.get(key) or ""
            else:
                row[key] = job.get(key)
    return row


def save_job_to_db(
    job: dict,
    db_client: PgVectorClient | None = None,
) -> None:
    """
    Insert or update a single job in the jobs table.

    Args:
        job: Job as dict (e.g. JobData or dataclass __dict__) with id, link, title, company, posted_date, etc.
        db_client: Optional PgVectorClient instance.
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    _ensure_jobs_table_exists(db_client)
    row = _job_to_row(job)
    with db_client:
        db_client.create_or_update_row(DEFAULT_JOBS_TABLE, row)
    logger.debug(f"Saved job {row['id']} to table '{DEFAULT_JOBS_TABLE}'.")


def load_jobs_list(
    db_client: PgVectorClient | None = None,
    order_by_posted_date_desc: bool = True,
) -> list[JobData]:
    """
    Load all jobs from the jobs table as a list of JobData dicts.

    Args:
        db_client: Optional PgVectorClient instance.
        order_by_posted_date_desc: If True, order by posted_date descending.

    Returns:
        List of JobData dictionaries (one per job).
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    _ensure_jobs_table_exists(db_client)
    with db_client:
        rows = db_client.get_rows(
            DEFAULT_JOBS_TABLE,
            order_by=("posted_date", "DESC") if order_by_posted_date_desc else None,
        )
    # Return only job-data keys (exclude created_at, updated_at) and ensure list/dict defaults
    jobs: list[JobData] = []
    for r in rows:
        job = {k: r.get(k) for k in JOBS_TABLE_DATA_KEYS if k in r}
        if not job.get("keywords"):
            job["keywords"] = []
        if not job.get("entities"):
            job["entities"] = {}
        if not job.get("tags"):
            job["tags"] = []
        if not job.get("meta"):
            job["meta"] = {}
        jobs.append(cast(JobData, job))
    return jobs


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
        jobs: list[JobData] = db_client.get_rows(DEFAULT_TABLE_DATA_NAME, ids=chunk_ids)

    return jobs


def load_jobs_embeddings(
    chunk_ids: list[str] | None = None, db_client: PgVectorClient | None = None
) -> dict[str, NDArray[np.float64]]:
    """
    Load job embeddings for given chunk IDs or all embeddings if no IDs provided.

    Args:
        chunk_ids: Optional list of chunk IDs to retrieve embeddings for
        db_client: Optional PgVectorClient instance

    Returns:
        Dictionary mapping chunk IDs to their embedding vectors as numpy arrays
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        embeddings = db_client.get_embeddings(DEFAULT_TABLE_NAME, chunk_ids)

        return embeddings


def generate_job_embeddings(
    data: list[JobData], embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL
) -> np.ndarray:
    embedding_dim = LLAMACPP_MODEL_EMBEDDING_SIZES[embed_model]
    if not data:
        return np.empty((0, embedding_dim), dtype=np.float32)

    texts = [f"{d['title']}\n{d['details']}" for d in data]

    embedder = LlamacppEmbedding(
        model=embed_model,
        use_cache=True,
        use_dynamic_batch_sizing=True,
        verbose=False,
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


def save_job_embeddings(
    jobs: list[JobData],
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
    overwrite_db: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    # Initialize database client if not provided
    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME, overwrite_db=overwrite_db
        )

    # Ensure both metadata and embeddings tables exist before fetching rows
    with db_client:
        # Create or verify metadata table
        metadata_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_DATA_NAME} (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            header_doc_id TEXT,
            parent_id TEXT,
            doc_index INTEGER,
            chunk_index INTEGER,
            num_tokens INTEGER,
            header TEXT,
            parent_header TEXT,
            content TEXT,
            level INTEGER,
            parent_level INTEGER,
            start_idx INTEGER,
            end_idx INTEGER,
            content_hash TEXT,
            text_hash TEXT,
            posted_date TIMESTAMPTZ
        );
        """
        with db_client.conn.cursor() as cur:
            cur.execute(metadata_table_query)
            logger.debug(
                f"Created or verified '{DEFAULT_TABLE_DATA_NAME}' table with 'content_hash', 'text_hash', and 'posted_date' columns."
            )

        # Create or verify embeddings table
        embedding_dimension = LLAMACPP_MODEL_EMBEDDING_SIZES[embed_model]
        db_client.create_table(DEFAULT_TABLE_NAME, dimension=embedding_dimension)
        logger.debug(
            f"Created or verified '{DEFAULT_TABLE_NAME}' table with dimension {embedding_dimension}."
        )

        # Fetch existing job metadata with hashes, filtering by doc_id
        existing_jobs = db_client.get_rows(
            DEFAULT_TABLE_DATA_NAME,
            ids=[compute_job_hash(job) for job in jobs],
            id_column="content_hash",
        )
        existing_job_hashes = {
            row["doc_id"]: row.get("content_hash") for row in existing_jobs
        }
        existing_text_hashes = {
            row["id"]: row.get("text_hash") for row in existing_jobs
        }
        logger.debug(f"Existing text hashes: {existing_text_hashes}")

    # Filter jobs that are new or have changed
    jobs_to_process: list[tuple[JobData, str]] = []
    for job in jobs:
        job_hash = compute_job_hash(job)
        existing_hash = existing_job_hashes.get(job["id"])
        if existing_hash is None or existing_hash != job_hash:
            jobs_to_process.append((job, job_hash))
        else:
            logger.debug(f"Skipping job {job['id']} - no changes detected.")

    # Sort jobs by posted_date (newest first)
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

    # Process only new or changed jobs
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
        if job.get("entities"):
            entities = job["entities"]
            if entities.get("role"):
                text += f"Role: {', '.join(entities['role'])}\n"
            if entities.get("technology_stack"):
                text += f"Technology Stack: {', '.join(entities['technology_stack'])}\n"
            if entities.get("qualifications"):
                text += f"Qualifications: {', '.join(entities['qualifications'])}\n"
            if entities.get("application"):
                text += f"Application: {', '.join(entities['application'])}\n"
        job_texts.append(text)

    # Rest of the original logic for chunking
    job_header_token_counts: list[int] = count_tokens(
        embed_model, job_headers, prevent_total=True
    )
    max_job_header_token = (
        max(job_header_token_counts) if job_header_token_counts else 0
    )
    chunks_with_data = chunk_texts_with_data(
        job_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_ids=[job["id"] for job, _ in jobs_to_process],
        buffer=max_job_header_token,
        model=embed_model,
    )
    for chunk in chunks_with_data:
        chunk["id"] = generate_key(
            chunk["doc_id"], chunk["chunk_index"], chunk["doc_index"]
        )
        logger.debug(
            f"Generated chunk ID: {chunk['id']} for doc_id: {chunk['doc_id']}, chunk_index: {chunk['chunk_index']}, doc_index: {chunk['doc_index']}"
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

    # Filter chunks that need new embeddings
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
            f"Chunk ID: {chunk['id']}, Computed text_hash: {text_hash}, Existing text_hash: {existing_text_hash}"
        )
        if existing_text_hash is None or existing_text_hash != text_hash:
            chunks_to_embed.append(chunk)
            embedding_texts.append(text)
            logger.info(
                f"Generating new embedding for chunk {chunk['id']} (new or changed content)"
            )
        else:
            # Retrieve existing embedding
            with db_client:
                embedding = db_client.get_embedding_by_id(
                    DEFAULT_TABLE_NAME, chunk["id"]
                )
                if embedding is not None:
                    existing_embeddings[chunk["id"]] = embedding
                    logger.info(
                        f"Reusing existing embedding for chunk {chunk['id']} from database"
                    )
                else:
                    logger.error(
                        f"No embedding found for unchanged chunk {chunk['id']}"
                    )
                    raise ValueError(
                        f"No embedding found for unchanged chunk {chunk['id']}"
                    )

    # Generate embeddings only for new or changed chunks
    # Use LlamacppEmbedding with caching and dynamic batch sizing to generate embeddings
    embedding_dimension = LLAMACPP_MODEL_EMBEDDING_SIZES[embed_model]
    if embedding_texts:
        embedder = LlamacppEmbedding(
            model=embed_model,
            use_cache=True,
            use_dynamic_batch_sizing=True,
            verbose=True,
        )
        new_embeddings = embedder.get_embeddings(
            embedding_texts,
            return_format="numpy",
            show_progress=True,
        )
    else:
        new_embeddings = np.empty((0, embedding_dimension), dtype=np.float32)
    if len(chunks_to_embed) != len(new_embeddings):
        raise ValueError(
            f"Mismatch between chunks_to_embed ({len(chunks_to_embed)}) and new_embeddings ({len(new_embeddings)})"
        )

    # Combine new and existing embeddings
    embeddings = []
    chunk_embedding_map = {
        chunks_to_embed[i]["id"]: new_embeddings[i] for i in range(len(chunks_to_embed))
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
        metadata = {
            key: value for key, value in job.items() if key not in ["title", "details"]
        }
        metadata["content_hash"] = job_hash
        row = {
            "id": chunk["id"],
            "metadata": metadata,
            "text": f"{job['title']}\n{chunk['content']}",
            "embedding": embedding,
            "content_hash": job_hash,
            "text_hash": chunk["text_hash"],
        }
        rows_data.append(row)
        required_keys = [
            "id",
            "doc_id",
            "doc_index",
            "chunk_index",
            "num_tokens",
            "content",
            "start_idx",
            "end_idx",
        ]
        missing_keys = [key for key in required_keys if key not in chunk]
        if missing_keys:
            logger.error(f"Missing keys in chunk: {missing_keys}")
            raise ValueError(f"Chunk missing required keys: {missing_keys}")
        header = job["title"]
        parent_header = job["company"]
        parent_id = generate_key(job["company"])
        header_doc_id = generate_key(job["title"])
        metadata_row = {
            "id": chunk["id"],
            "doc_id": chunk["doc_id"],
            "header_doc_id": header_doc_id,
            "parent_id": parent_id,
            "doc_index": chunk["doc_index"],
            "chunk_index": chunk["chunk_index"],
            "num_tokens": chunk["num_tokens"],
            "header": header,
            "parent_header": parent_header,
            "content": chunk["content"],
            "level": 1,
            "parent_level": 0,
            "start_idx": chunk["start_idx"],
            "end_idx": chunk["end_idx"],
            "content_hash": job_hash,
            "text_hash": chunk["text_hash"],
            "posted_date": job["posted_date"],
        }
        metadata_rows.append(metadata_row)

    with db_client:
        try:
            # Ensure metadata table includes content_hash, text_hash, and posted_date columns
            metadata_table_query = f"""
            CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_DATA_NAME} (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                header_doc_id TEXT,
                parent_id TEXT,
                doc_index INTEGER,
                chunk_index INTEGER,
                num_tokens INTEGER,
                header TEXT,
                parent_header TEXT,
                content TEXT,
                level INTEGER,
                parent_level INTEGER,
                start_idx INTEGER,
                end_idx INTEGER,
                content_hash TEXT,
                text_hash TEXT,
                posted_date TIMESTAMPTZ
            );
            """
            with db_client.conn.cursor() as cur:
                cur.execute(metadata_table_query)
                logger.debug(
                    f"Created or verified '{DEFAULT_TABLE_DATA_NAME}' table with 'content_hash', 'text_hash', and 'posted_date' columns."
                )

            # Log whether rows are created or updated for embeddings table
            with db_client.conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT id FROM {} WHERE id = ANY(%s)").format(
                        sql.Identifier(DEFAULT_TABLE_NAME)
                    ),
                    ([row["id"] for row in rows_data],),
                )
                existing_ids = {row["id"] for row in cur.fetchall()}
            create_count = sum(1 for row in rows_data if row["id"] not in existing_ids)
            update_count = len(rows_data) - create_count
            if create_count > 0:
                logger.info(
                    f"Creating {create_count} new rows in '{DEFAULT_TABLE_NAME}' table"
                )
            if update_count > 0:
                logger.info(
                    f"Updating {update_count} existing rows in '{DEFAULT_TABLE_NAME}' table"
                )

            results = db_client.create_or_update_rows(
                DEFAULT_TABLE_NAME,
                rows_data,
                dimension=embeddings.shape[1] if embeddings.size > 0 else None,
            )
            logger.success(
                f"Saved {len(results)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
            )

            # Log whether rows are created or updated for metadata table
            with db_client.conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT id FROM {} WHERE id = ANY(%s)").format(
                        sql.Identifier(DEFAULT_TABLE_DATA_NAME)
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
                    f"Creating {metadata_create_count} new rows in '{DEFAULT_TABLE_DATA_NAME}' table"
                )
            if metadata_update_count > 0:
                logger.info(
                    f"Updating {metadata_update_count} existing rows in '{DEFAULT_TABLE_DATA_NAME}' table"
                )

            logger.debug(f"Metadata rows to save: {len(metadata_rows)}")
            for idx, row in enumerate(metadata_rows):
                if "id" not in row:
                    logger.error(f"Row {idx} missing id: {row}")
                    raise ValueError(f"Row {idx} missing id")
            metadata_results = db_client.create_or_update_rows(
                DEFAULT_TABLE_DATA_NAME, metadata_rows
            )
            logger.success(
                f"Saved {len(metadata_results)} metadata records to '{DEFAULT_TABLE_DATA_NAME}' table."
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
    embedder = LlamacppEmbedding(
        model=embed_model,
        use_cache=True,
        use_dynamic_batch_sizing=False,
        verbose=False,
    )
    query_embedding = embedder.get_embeddings(
        [query],
        return_format="numpy",
        show_progress=False,
    )[0]

    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        results = db_client.search(
            table_name=DEFAULT_TABLE_NAME,
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold,
        )

        return results
