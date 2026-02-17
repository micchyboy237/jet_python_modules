from typing import cast

import numpy as np
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_EMBEDDING_SIZES
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.adapters.llama_cpp.utils import get_context_size
from jet.data.utils import generate_hash
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger
from jet.vectors.reranker.bm25 import rerank_bm25
from jet.wordnet.text_chunker import truncate_texts_fast
from numpy.typing import NDArray
from psycopg import sql
from shared.data_types.job import JobData, JobSearchResult

DEFAULT_EMBED_MODEL: LLAMACPP_EMBED_KEYS = "nomic-embed-text-v2-moe"
DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
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
    ids: list[str] | None = None,
) -> list[JobData]:
    """
    Load all jobs from the jobs table as a list of JobData dicts.

    Args:
        db_client: Optional PgVectorClient instance.
        order_by_posted_date_desc: If True, order by posted_date descending.
        ids: Optional list of job IDs to retrieve; if None, loads all jobs.

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
            ids=ids,
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
        jobs: list[JobData] = db_client.get_rows(DEFAULT_TABLE_NAME, ids=chunk_ids)

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
    # 🔥 Use actual model context size instead of DEFAULT_CHUNK_SIZE

    embedding_dim = LLAMACPP_MODEL_EMBEDDING_SIZES[embed_model]
    if not data:
        return np.empty((0, embedding_dim), dtype=np.float32)

    texts = [f"{d['title']}\n{d['details']}" for d in data]

    model_ctx_size = get_context_size(embed_model)
    # Leave safety margin (llama.cpp can be strict)
    safe_max_tokens = max(1, model_ctx_size - 8)

    truncated_texts = truncate_texts_fast(
        texts, embed_model, safe_max_tokens, strict_sentences=True
    )

    embedder = LlamacppEmbedding(
        model=embed_model,
        use_cache=True,
        use_dynamic_batch_sizing=True,
        verbose=False,
    )
    embeddings = embedder.get_embeddings(
        truncated_texts,
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
) -> dict[str, str]:
    """
    Save job embeddings to the database and return a mapping from job ID to embedding row ID.

    Args:
        jobs: List of JobData entries to embed and save.
        embed_model: The embedding model identifier.
        db_client: The PgVectorClient instance; if None, one will be created.
        overwrite_db: Whether to overwrite the database on creation.
        chunk_size: Not currently used (future: chunk support).
        chunk_overlap: Not currently used (future: chunk support).

    Returns:
        Dict[str, str]: Mapping from job.id to embedding row id.
    """
    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME, overwrite_db=overwrite_db
        )

    if not jobs:
        return {}

    # Prepare texts for embedding
    texts = [f"{job['title']}\n{job.get('details', '')}".strip() for job in jobs]

    # Generate embeddings
    embeddings_array: np.ndarray = generate_job_embeddings(jobs, embed_model)

    # Prepare rows for embeddings table
    embedding_rows = []
    id_mapping: dict[str, str] = {}  # job.id → embedding row id

    for idx, (job, emb_vector) in enumerate(zip(jobs, embeddings_array)):
        emb_id = generate_hash(str(job["id"]) + "_embedding")
        row = {
            "id": emb_id,
            "job_id": job["id"],  # link to jobs table
            "embedding": emb_vector.tolist(),
            "text": texts[idx],  # for debug / hybrid search
            "title": job["title"],
            "company": job.get("company"),
            "link": job["link"],
            "posted_date": job["posted_date"],
        }
        embedding_rows.append(row)
        id_mapping[job["id"]] = emb_id

    dimension = embeddings_array.shape[1]

    # Bulk upsert embeddings
    db_client.create_or_update_rows(
        table_name=DEFAULT_TABLE_NAME,
        rows_data=embedding_rows,
        dimension=dimension,
    )

    logger.success(
        f"Saved {len(jobs)} job embeddings into table '{DEFAULT_TABLE_NAME}'"
    )
    return id_mapping


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
        show_progress=True,
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


def hybrid_search_jobs(
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
    embed_model: LLAMACPP_EMBED_KEYS = DEFAULT_EMBED_MODEL,
    db_client: PgVectorClient | None = None,
) -> list[JobSearchResult]:
    raw_results = search_jobs(
        query=query,
        top_k=top_k,
        threshold=threshold,
        embed_model=embed_model,
        db_client=db_client,
    )

    ids = [result["id"] for result in raw_results]
    documents = [result["text"] for result in raw_results]
    query_candidates, reranked_results = rerank_bm25(query, documents, ids)

    return reranked_results
