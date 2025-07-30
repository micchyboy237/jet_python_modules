from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from jet.wordnet.text_chunker import chunk_texts, truncate_texts
from shared.data_types.job import JobData, JobMetadata, JobSearchResult
from jet.db.postgres.pgvector import PgVectorClient, EmbeddingInput, SearchResult
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, DocChunkResult
from jet.models.tokenizer.base import TokenizerWrapper

DEFAULT_EMBED_MODEL: EmbedModelType = "mxbai-embed-large"
DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_TABLE_DATA_NAME = f"{DEFAULT_TABLE_NAME}_data"
DEFAULT_CHUNK_SIZE = 512


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
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None,
    overwrite_db: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict:
    """
    Save job embeddings to the database after chunking the job descriptions.

    Args:
        jobs: List of job data dictionaries
        embed_model: Embedding model to use
        db_client: Optional database client
        overwrite_db: Whether to overwrite the database
        chunk_size: Maximum number of tokens per chunk
    """
    # Existing code for chunking and embedding generation
    job_texts = [f"# {job['title']}\n{job['details']}" for job in jobs]
    chunks = chunk_texts(
        job_texts,
        chunk_size=chunk_size,
        ids=[job["id"] for job in jobs]
    )

    # chunk_texts = [
    #     f"{chunk['header']}\n{chunk['content']}" for chunk in chunks]

    job_texts = [f"# {job['title']}\n{job['details']}" for job in jobs]
    truncated_job_texts = truncate_texts(job_texts, embed_model, chunk_size)
    embeddings = generate_embeddings(
        truncated_job_texts,
        embed_model,
        show_progress=True,
        return_format="numpy"
    )

    # if len(chunks) != len(embeddings):
    #     raise ValueError(
    #         f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
    #     )
    if len(job_texts) != len(embeddings):
        raise ValueError(
            f"Mismatch between job_texts ({len(job_texts)}) and embeddings ({len(embeddings)})"
        )

    rows_data = [
        {
            # **chunk,
            **job,
            "embedding": embedding
        }
        # for chunk, embedding in zip(chunks, embeddings)
        for job, embedding in zip(jobs, embeddings)
    ]

    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME,
            overwrite_db=overwrite_db
        )

    logger.info(
        f"Inserting {len(rows_data)} chunked job embeddings into '{DEFAULT_TABLE_NAME}' table..."
    )
    with db_client:
        db_client.create_rows(DEFAULT_TABLE_NAME, rows_data)
        # Store chunk metadata in a separate table
        metadata_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_DATA_NAME} (
            chunk_id TEXT PRIMARY KEY,
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
            end_idx INTEGER
        );
        """
        try:
            with db_client.conn.cursor() as cur:
                cur.execute(metadata_query)
                logger.info(
                    f"Created or verified '{DEFAULT_TABLE_DATA_NAME}' table.")
                for chunk in chunks:
                    # Validate chunk metadata
                    required_keys = [
                        "id", "doc_id", "header_doc_id", "parent_id", "doc_index",
                        "chunk_index", "num_tokens", "header", "parent_header",
                        "content", "level", "parent_level", "metadata"
                    ]
                    missing_keys = [
                        key for key in required_keys if key not in chunk]
                    if missing_keys:
                        logger.error(f"Missing keys in chunk: {missing_keys}")
                        raise ValueError(
                            f"Chunk missing required keys: {missing_keys}")
                    if "start_idx" not in chunk["metadata"] or "end_idx" not in chunk["metadata"]:
                        logger.error(
                            f"Chunk metadata missing start_idx or end_idx: {chunk['id']}")
                        raise ValueError(
                            f"Chunk metadata missing start_idx or end_idx for chunk {chunk['id']}")

                    cur.execute(
                        f"""
                        INSERT INTO {DEFAULT_TABLE_DATA_NAME} (
                            chunk_id, doc_id, header_doc_id, parent_id, doc_index, chunk_index,
                            num_tokens, header, parent_header, content, level, parent_level,
                            start_idx, end_idx
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING;
                        """,
                        (
                            chunk["id"],
                            chunk["doc_id"],
                            chunk["header_doc_id"],
                            chunk["parent_id"],
                            chunk["doc_index"],
                            chunk["chunk_index"],
                            chunk["num_tokens"],
                            chunk["header"],
                            chunk["parent_header"],
                            chunk["content"],
                            chunk["level"],
                            chunk["parent_level"],
                            chunk["metadata"]["start_idx"],
                            chunk["metadata"]["end_idx"]
                        )
                    )
                db_client.conn.commit()  # Explicitly commit the transaction
                logger.info(
                    f"Inserted {len(chunks)} metadata records into '{DEFAULT_TABLE_DATA_NAME}' table.")
        except Exception as e:
            logger.error(f"Failed to insert metadata: {str(e)}")
            db_client.conn.rollback()  # Rollback on error
            raise
        logger.success(
            f"Saved {len(rows_data)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
        )
        return {
            "job_texts": job_texts,
            "truncated_job_texts": truncated_job_texts,
            "embeddings": embeddings,
            "rows": rows_data,
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
