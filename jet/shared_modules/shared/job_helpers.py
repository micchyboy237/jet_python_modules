import math
from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from jet.data.utils import generate_hash, generate_key
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from jet.wordnet.text_chunker import chunk_texts_with_data, truncate_texts
from shared.data_types.job import JobData, JobMetadata, JobSearchResult
from jet.db.postgres.pgvector import PgVectorClient, EmbeddingInput, SearchResult
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, DocChunkResult
from jet.models.tokenizer.base import TokenizerWrapper, count_tokens

DEFAULT_EMBED_MODEL: EmbedModelType = "mxbai-embed-large"
DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_TABLE_DATA_NAME = f"{DEFAULT_TABLE_NAME}_data"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def get_jobs_db_summary(db_client: Optional[PgVectorClient] = None):
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)
    db_summary = db_client.get_database_summary()
    return db_summary


def load_jobs(
    chunk_ids: Optional[List[str]] = None,
    db_client: Optional[PgVectorClient] = None
) -> List[JobMetadata]:
    """
    Load job job for given chunk IDs or all job if no IDs provided.

    Args:
        chunk_ids: Optional list of chunk IDs to retrieve job for
        db_client: Optional PgVectorClient instance

    Returns:
        List of JobMetadata dictionaries containing job job
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    job: List[JobMetadata] = []
    with db_client:
        with db_client.conn.cursor() as cur:
            if chunk_ids:
                query = f"""
                    SELECT 
                        chunk_id, doc_id, header_doc_id, parent_id, doc_index,
                        chunk_index, num_tokens, header, parent_header, content,
                        level, parent_level, start_idx, end_idx
                    FROM {DEFAULT_TABLE_DATA_NAME} 
                    WHERE chunk_id = ANY(%s);
                """
                cur.execute(query, (chunk_ids,))
            else:
                query = f"""
                    SELECT 
                        chunk_id, doc_id, header_doc_id, parent_id, doc_index,
                        chunk_index, num_tokens, header, parent_header, content,
                        level, parent_level, start_idx, end_idx
                    FROM {DEFAULT_TABLE_DATA_NAME};
                """
                cur.execute(query)

            rows = cur.fetchall()
            for row in rows:
                job.append({
                    "id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "header_doc_id": row["header_doc_id"],
                    "parent_id": row["parent_id"],
                    "doc_index": row["doc_index"],
                    "chunk_index": row["chunk_index"],
                    "num_tokens": row["num_tokens"],
                    "header": row["header"],
                    "parent_header": row["parent_header"],
                    "content": row["content"],
                    "level": row["level"],
                    "parent_level": row["parent_level"],
                    "start_idx": row["start_idx"],
                    "end_idx": row["end_idx"]
                })

    return job


def load_jobs_embeddings(
    chunk_ids: Optional[List[str]] = None,
    db_client: Optional[PgVectorClient] = None
) -> Dict[str, NDArray[np.float64]]:
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


def generate_job_embeddings(data: List[JobData], embed_model: EmbedModelType = DEFAULT_EMBED_MODEL) -> np.ndarray:
    texts = [f"{d['title']}\n{d['details']}" for d in data]
    embeddings = generate_embeddings(
        texts, embed_model, show_progress=True, return_format="numpy")
    assert isinstance(
        embeddings, np.ndarray), f"Expected np.ndarray, got {type(embeddings)}"
    return embeddings


def save_job_embeddings(
    jobs: List[JobData],
    embed_model: str = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None,
    overwrite_db: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    def compute_job_hash(job: JobData) -> str:
        """Compute a hash of the job's content (excluding ID) using hash_text."""
        job_copy = {k: v for k, v in job.items() if k != 'id'}
        return generate_hash(job_copy)

    # Initialize database client if not provided
    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME,
            overwrite_db=overwrite_db
        )

    # Fetch existing job metadata with hashes
    with db_client:
        existing_jobs = db_client.get_rows(DEFAULT_TABLE_DATA_NAME, ids=[
                                           job['id'] for job in jobs])
        existing_job_hashes = {row['doc_id']: row.get(
            'content_hash') for row in existing_jobs}

    # Filter jobs that are new or have changed
    jobs_to_process = []
    for job in jobs:
        job_hash = compute_job_hash(job)
        existing_hash = existing_job_hashes.get(job['id'])
        if existing_hash is None or existing_hash != job_hash:
            jobs_to_process.append((job, job_hash))
        else:
            logger.debug(f"Skipping job {job['id']} - no changes detected.")

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
        job_by_id[job['id']] = job
        header = f"{job['title']}"
        job_headers.append(header)
        text = ""
        text += f"Details\n{job['details']}\n\n"
        text += f"Company: {job['company']}\n"
        if job.get('keywords'):
            text += f"Keywords: {', '.join(job['keywords'])}\n"
        if job.get('job_type'):
            text += f"Job Type: {job['job_type']}\n"
        if job.get('salary'):
            text += f"Salary: {job['salary']}\n"
        if job.get('hours_per_week'):
            text += f"Hours per Week: {job['hours_per_week']}\n"
        if job.get('entities'):
            entities = job['entities']
            if entities.get('role'):
                text += f"Role: {', '.join(entities['role'])}\n"
            if entities.get('technology_stack'):
                text += f"Technology Stack: {', '.join(entities['technology_stack'])}\n"
            if entities.get('qualifications'):
                text += f"Qualifications: {', '.join(entities['qualifications'])}\n"
            if entities.get('application'):
                text += f"Application: {', '.join(entities['application'])}\n"
        job_texts.append(text)

    # Rest of the original logic for chunking and embedding
    job_header_token_counts: List[int] = count_tokens(
        embed_model, job_headers, prevent_total=True)
    max_job_header_token = max(
        job_header_token_counts) if job_header_token_counts else 0
    chunks_with_data = chunk_texts_with_data(
        job_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_ids=[job['id'] for job, _ in jobs_to_process],
        buffer=max_job_header_token,
        model=embed_model,
    )
    for chunk in chunks_with_data:
        chunk["id"] = generate_key(
            chunk["doc_id"], chunk["chunk_index"], chunk["doc_index"]
        )
        logger.debug(
            f"Generated chunk ID: {chunk['id']} for doc_id: {chunk['doc_id']}, chunk_index: {chunk['chunk_index']}, doc_index: {chunk['doc_index']}")

    all_num_tokens = [chunk["num_tokens"] for chunk in chunks_with_data]
    count = len(chunks_with_data)
    min_token = min(all_num_tokens) if all_num_tokens else 0
    ave_token = math.ceil(sum(all_num_tokens) /
                          len(all_num_tokens)) if all_num_tokens else 0
    max_token = max(all_num_tokens) if all_num_tokens else 0
    logger.log("count:", count, colors=["GRAY", "INFO"])
    logger.log("min_token:", min_token, colors=["GRAY", "SUCCESS"])
    logger.log("ave_token:", ave_token, colors=["GRAY", "SUCCESS"])
    logger.log("max_token:", max_token, colors=["GRAY", "SUCCESS"])

    embedding_texts = []
    for chunk in chunks_with_data:
        job = job_by_id.get(chunk["doc_id"])
        if not job:
            logger.error(f"No job found for doc_id: {chunk['doc_id']}")
            raise ValueError(f"No job found for doc_id: {chunk['doc_id']}")
        header = f"{job['title']}"
        text = f"{header}\n{chunk['content']}"
        embedding_texts.append(text)

    embeddings = generate_embeddings(
        embedding_texts,
        embed_model,
        show_progress=True,
        return_format="numpy"
    )
    if len(chunks_with_data) != len(embeddings):
        raise ValueError(
            f"Mismatch between chunks_with_data ({len(chunks_with_data)}) and embeddings ({len(embeddings)})"
        )

    rows_data = []
    metadata_rows = []
    for text, chunk, embedding in zip(embedding_texts, chunks_with_data, embeddings):
        job, job_hash = next(
            (j, h) for j, h in jobs_to_process if j['id'] == chunk["doc_id"])
        metadata = {
            key: value for key, value in job.items()
            if key not in ['title', 'details']
        }
        metadata['content_hash'] = job_hash
        row = {
            "id": chunk["id"],
            "metadata": metadata,
            "text": text,
            "embedding": embedding
        }
        rows_data.append(row)
        required_keys = [
            "id", "doc_id", "doc_index", "chunk_index", "num_tokens",
            "content", "start_idx", "end_idx"
        ]
        missing_keys = [
            key for key in required_keys if key not in chunk]
        if missing_keys:
            logger.error(f"Missing keys in chunk: {missing_keys}")
            raise ValueError(
                f"Chunk missing required keys: {missing_keys}")
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
            "content_hash": job_hash
        }
        metadata_rows.append(metadata_row)

    with db_client:
        try:
            # Ensure metadata table includes content_hash column
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
                content_hash TEXT
            );
            """
            with db_client.conn.cursor() as cur:
                cur.execute(metadata_table_query)
                logger.debug(
                    f"Created or verified '{DEFAULT_TABLE_DATA_NAME}' table with 'content_hash' column.")

            results = db_client.create_or_update_rows(
                DEFAULT_TABLE_NAME,
                rows_data,
                dimension=embeddings.shape[1] if embeddings.size > 0 else None
            )
            logger.success(
                f"Saved {len(results)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
            )
            logger.debug(f"Metadata rows to save: {len(metadata_rows)}")
            for idx, row in enumerate(metadata_rows):
                if "id" not in row:
                    logger.error(f"Row {idx} missing id: {row}")
                    raise ValueError(f"Row {idx} missing id")
            metadata_results = db_client.create_or_update_rows(
                DEFAULT_TABLE_DATA_NAME,
                metadata_rows
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
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None
) -> List[JobSearchResult]:
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
    query_embedding = generate_embeddings(
        [query],
        embed_model,
        show_progress=False,
        return_format="numpy"
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
