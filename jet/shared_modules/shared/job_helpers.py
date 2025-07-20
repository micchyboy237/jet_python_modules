from typing import List, Dict, Optional
import numpy as np
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from shared.data_types.job import JobData
from jet.db.postgres.pgvector import PgVectorClient, VectorInput, SearchResult
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, DocChunkResult
from jet.models.tokenizer.base import TokenizerWrapper

DEFAULT_EMBED_MODEL: EmbedModelType = "mxbai-embed-large"
DEFAULT_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_CHUNK_SIZE = 512


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
    tokenizer: Optional[TokenizerWrapper] = None
) -> None:
    """
    Save job embeddings to the database after chunking the job descriptions.

    Args:
        jobs: List of job data dictionaries
        embed_model: Embedding model to use
        db_client: Optional database client
        overwrite_db: Whether to overwrite the database
        chunk_size: Maximum number of tokens per chunk
        tokenizer: Optional tokenizer for chunking
    """
    # Generate chunks for all job descriptions
    job_texts = [f"# {job['title']}\n{job['details']}" for job in jobs]
    chunks = chunk_docs_by_hierarchy(
        markdown_texts=job_texts,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
        ids=[job["id"] for job in jobs]
    )

    # Generate embeddings for all chunks
    chunk_texts = [
        f"{chunk['header']}\n{chunk['content']}" for chunk in chunks]
    embeddings = generate_embeddings(
        chunk_texts,
        embed_model,
        show_progress=True,
        return_format="numpy"
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
        )

    # Prepare vector data with chunk IDs
    vector_data: Dict[str, VectorInput] = {
        chunk["id"]: embedding for chunk, embedding in zip(chunks, embeddings)
    }

    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_DB_NAME,
            overwrite_db=overwrite_db
        )

    logger.info(
        f"Inserting {len(vector_data)} chunked job embeddings into '{DEFAULT_TABLE_NAME}' table..."
    )
    with db_client:
        db_client.insert_vector_by_ids(DEFAULT_TABLE_NAME, vector_data)
        # Store chunk metadata in a separate table
        metadata_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_NAME}_metadata (
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
        with db_client.conn.cursor() as cur:
            cur.execute(metadata_query)
            for chunk in chunks:
                cur.execute(
                    f"""
                    INSERT INTO {DEFAULT_TABLE_NAME}_metadata (
                        chunk_id, doc_id, header_doc_id, parent_id, doc_index, chunk_index,
                        num_tokens, header, parent_header, content, level, parent_level,
                        start_idx, end_idx
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
        logger.success(
            f"Saved {len(vector_data)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
        )


def search_jobs(
    query: str,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None
) -> List[Dict]:
    """
    Search for jobs based on a query string using vector similarity.

    Args:
        query: Search query string
        top_k: Number of results to return
        embed_model: Embedding model to use for query
        db_client: Optional database client

    Returns:
        List of dictionaries containing job IDs and similarity scores
    """
    logger.debug("Starting job search with query: %s", query)

    # Generate embedding for the query
    query_embedding = generate_embeddings(
        [query],
        embed_model,
        show_progress=False,
        return_format="numpy"
    )[0]

    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_DB_NAME)

    # Search for similar vectors
    with db_client:
        # Ensure metadata table exists using a custom query
        metadata_query = f"""
        CREATE TABLE IF NOT EXISTS {DEFAULT_TABLE_NAME}_metadata (
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
        with db_client.conn.cursor() as cur:
            logger.debug("Ensuring metadata table exists")
            cur.execute(metadata_query)
            cur.execute(
                f"SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s;",
                (f"{DEFAULT_TABLE_NAME}_metadata",)
            )
            table_exists = cur.fetchone()
            logger.debug("Metadata table exists: %s", bool(table_exists))
            if not table_exists:
                logger.warning(
                    "Metadata table was created during search, indicating potential data inconsistency")

        results = db_client.search_similar(
            table_name=DEFAULT_TABLE_NAME,
            query_vector=query_embedding,
            top_k=top_k * 2  # Get extra results to account for multiple chunks per job
        )

        # Aggregate results by doc_id
        job_results: Dict[str, Dict] = {}
        with db_client.conn.cursor() as cur:
            for result in results:
                logger.debug(
                    "Querying metadata for chunk_id: %s", result["id"])
                cur.execute(
                    f"SELECT doc_id FROM {DEFAULT_TABLE_NAME}_metadata WHERE chunk_id = %s;",
                    (result["id"],)
                )
                row = cur.fetchone()
                if row:
                    doc_id = row["doc_id"]
                    logger.debug("Found chunk_id %s with doc_id %s",
                                 result["id"], doc_id)
                    if doc_id not in job_results:
                        job_results[doc_id] = {
                            "id": doc_id,
                            "score": result["score"],
                            "chunk_count": 1
                        }
                    else:
                        job_results[doc_id]["score"] = max(
                            job_results[doc_id]["score"], result["score"]
                        )
                        job_results[doc_id]["chunk_count"] += 1
                else:
                    logger.warning(
                        "No metadata found for chunk_id: %s", result["id"])

        # Sort by score and limit to top_k
        final_results = sorted(
            job_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        logger.debug("Found %d unique jobs in search results",
                     len(final_results))
        # Remove chunk_count from final output
        return [
            {"id": result["id"], "score": result["score"]}
            for result in final_results
        ]
