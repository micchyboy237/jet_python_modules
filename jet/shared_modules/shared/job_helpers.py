from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from shared.data_types.job import JobData, JobMetadata, JobSearchResult
from jet.db.postgres.pgvector import PgVectorClient, VectorInput, SearchResult
from jet.models.embeddings.chunking import chunk_docs_by_hierarchy, DocChunkResult
from jet.models.tokenizer.base import TokenizerWrapper

DEFAULT_EMBED_MODEL: EmbedModelType = "mxbai-embed-large"
DEFAULT_JOBS_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_CHUNK_SIZE = 512


def load_job_embeddings(
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

    embeddings: Dict[str, NDArray[np.float64]] = {}
    with db_client:
        with db_client.conn.cursor() as cur:
            if chunk_ids:
                query = f"""
                    SELECT id, embedding
                    FROM {DEFAULT_TABLE_NAME}
                    WHERE id = ANY(%s);
                """
                cur.execute(query, (chunk_ids,))
            else:
                query = f"""
                    SELECT id, embedding
                    FROM {DEFAULT_TABLE_NAME};
                """
                cur.execute(query)

            rows = cur.fetchall()
            for row in rows:
                embeddings[row["id"]] = np.array(
                    row["embedding"], dtype=np.float64)

    logger.debug("Loaded %d job embeddings", len(embeddings))
    return embeddings


def load_jobs_metadata(
    chunk_ids: Optional[List[str]] = None,
    db_client: Optional[PgVectorClient] = None
) -> List[JobMetadata]:
    """
    Load job metadata for given chunk IDs or all metadata if no IDs provided.

    Args:
        chunk_ids: Optional list of chunk IDs to retrieve metadata for
        db_client: Optional PgVectorClient instance

    Returns:
        List of JobMetadata dictionaries containing job metadata
    """
    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    metadata: List[JobMetadata] = []
    with db_client:
        with db_client.conn.cursor() as cur:
            if chunk_ids:
                query = f"""
                    SELECT 
                        chunk_id, doc_id, header_doc_id, parent_id, doc_index,
                        chunk_index, num_tokens, header, parent_header, content,
                        level, parent_level, start_idx, end_idx
                    FROM {DEFAULT_TABLE_NAME}_metadata 
                    WHERE chunk_id = ANY(%s);
                """
                cur.execute(query, (chunk_ids,))
            else:
                query = f"""
                    SELECT 
                        chunk_id, doc_id, header_doc_id, parent_id, doc_index,
                        chunk_index, num_tokens, header, parent_header, content,
                        level, parent_level, start_idx, end_idx
                    FROM {DEFAULT_TABLE_NAME}_metadata;
                """
                cur.execute(query)

            rows = cur.fetchall()
            for row in rows:
                metadata.append({
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

    logger.debug("Loaded metadata for %d chunks", len(metadata))
    return metadata


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
    # Existing code for chunking and embedding generation
    job_texts = [f"# {job['title']}\n{job['details']}" for job in jobs]
    chunks = chunk_docs_by_hierarchy(
        markdown_texts=job_texts,
        chunk_size=chunk_size,
        tokenizer=tokenizer,
        ids=[job["id"] for job in jobs]
    )

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

    vector_data: Dict[str, VectorInput] = {
        chunk["id"]: embedding for chunk, embedding in zip(chunks, embeddings)
    }

    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_JOBS_DB_NAME,
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
        try:
            with db_client.conn.cursor() as cur:
                cur.execute(metadata_query)
                logger.info(
                    f"Created or verified '{DEFAULT_TABLE_NAME}_metadata' table.")
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
                        INSERT INTO {DEFAULT_TABLE_NAME}_metadata (
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
                    f"Inserted {len(chunks)} metadata records into '{DEFAULT_TABLE_NAME}_metadata' table.")
        except Exception as e:
            logger.error(f"Failed to insert metadata: {str(e)}")
            db_client.conn.rollback()  # Rollback on error
            raise
        logger.success(
            f"Saved {len(vector_data)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
        )


def search_jobs(
    query: str,
    top_k: int = 5,
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None
) -> List[JobSearchResult]:
    """
    Search for jobs based on a query string and return ranked results with metadata.

    Args:
        query: Search query string
        top_k: Number of top results to return
        embed_model: Embedding model to use
        db_client: Optional PgVectorClient instance

    Returns:
        List of JobSearchResult dictionaries containing rank, score, and job metadata
    """
    logger.debug("Starting job search with query: %s", query)
    query_embedding = generate_embeddings(
        [query],
        embed_model,
        show_progress=False,
        return_format="numpy"
    )[0]

    if not db_client:
        db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    with db_client:
        with db_client.conn.cursor() as cur:
            logger.debug("Ensuring metadata table exists")
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
            top_k=top_k * 2
        )
        logger.debug("Retrieved %d similar results: %s",
                     len(results), [r["id"] for r in results])

        chunk_ids = [result["id"] for result in results]
        metadata_list = load_jobs_metadata(chunk_ids, db_client)
        logger.debug("Retrieved %d metadata records: %s", len(
            metadata_list), [m["id"] for m in metadata_list])

        # Create a lookup dictionary for metadata to ensure order and matching
        metadata_dict = {m["id"]: m for m in metadata_list}

        job_results: Dict[str, Dict] = {}
        for result in results:
            chunk_id = result["id"]
            metadata = metadata_dict.get(chunk_id)
            if not metadata:
                logger.warning("No metadata found for chunk_id: %s", chunk_id)
                continue

            if metadata["id"] != chunk_id:
                logger.error(
                    "Metadata ID %s does not match result ID %s", metadata["id"], chunk_id)
                continue

            doc_id = metadata["doc_id"]
            if doc_id not in job_results:
                job_results[doc_id] = {
                    "score": result["score"],
                    "chunk_count": 1,
                    **metadata
                }
            else:
                job_results[doc_id]["score"] = max(
                    job_results[doc_id]["score"], result["score"]
                )
                job_results[doc_id]["chunk_count"] += 1

        final_results = sorted(
            job_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        logger.debug("Found %d unique jobs in search results",
                     len(final_results))

        return [
            {
                "rank": idx + 1,
                "score": result["score"],
                "id": result["id"],
                "doc_id": result["doc_id"],
                "header_doc_id": result["header_doc_id"],
                "parent_id": result["parent_id"],
                "doc_index": result["doc_index"],
                "chunk_index": result["chunk_index"],
                "num_tokens": result["num_tokens"],
                "header": result["header"],
                "parent_header": result["parent_header"],
                "content": result["content"],
                "level": result["level"],
                "parent_level": result["parent_level"],
                "start_idx": result["start_idx"],
                "end_idx": result["end_idx"]
            }
            for idx, result in enumerate(final_results)
        ]
