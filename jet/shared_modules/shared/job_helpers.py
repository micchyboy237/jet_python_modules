import math
from typing import List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
from jet.data.utils import generate_key
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
    embed_model: EmbedModelType = DEFAULT_EMBED_MODEL,
    db_client: Optional[PgVectorClient] = None,
    overwrite_db: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    """
    Save job embeddings to the database after chunking the job descriptions.

    Args:
        jobs: List of job data dictionaries
        embed_model: Embedding model to use
        db_client: Optional database client
        overwrite_db: Whether to overwrite the database
        chunk_size: Maximum number of tokens per chunk
        chunk_overlap: Number of tokens to overlap between consecutive chunks

    Returns:
        Dictionary containing job_texts, embeddings, and rows inserted
    """
    # Enhanced job_texts for applicant-focused RAG search
    job_headers = []
    job_texts = []
    for job in jobs:
        header = f"# {job['title']}\n"
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

    job_header_token_counts: List[int] = count_tokens(
        embed_model, job_headers, prevent_total=True)
    max_job_header_token = max(job_header_token_counts)

    chunks_with_data = chunk_texts_with_data(
        job_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_ids=[job['id'] for job in jobs],  # Use job['id'] as doc_id
        buffer=max_job_header_token,
        model=embed_model,
    )
    all_num_tokens = [chunk["num_tokens"] for chunk in chunks_with_data]
    count = len(chunks_with_data)
    min_token = min(all_num_tokens)
    ave_token = math.ceil(sum(all_num_tokens) /
                          len(all_num_tokens)) if all_num_tokens else 0
    max_token = max(all_num_tokens)

    logger.log("count:", count, colors=["GRAY", "INFO"])
    logger.log("min_token:", min_token, colors=["GRAY", "SUCCESS"])
    logger.log("ave_token:", ave_token, colors=["GRAY", "SUCCESS"])
    logger.log("max_token:", max_token, colors=["GRAY", "SUCCESS"])

    # Generate embeddings for each chunk, prepending header for chunks with chunk_index > 0
    # Create lookup for job titles
    job_by_id = {job['id']: job for job in jobs}
    embedding_texts = []
    for chunk in chunks_with_data:
        job = job_by_id.get(chunk["doc_id"])
        if not job:
            logger.error(f"No job found for doc_id: {chunk['doc_id']}")
            raise ValueError(f"No job found for doc_id: {chunk['doc_id']}")
        header = f"# {job['title']}"
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

    # Create a lookup for jobs by id
    job_by_id = {job['id']: job for job in jobs}
    rows_data = []
    for text, chunk, embedding in zip(embedding_texts, chunks_with_data, embeddings):
        job = job_by_id.get(chunk["doc_id"])
        if not job:
            logger.error(f"No job found for doc_id: {chunk['doc_id']}")
            raise ValueError(f"No job found for doc_id: {chunk['doc_id']}")

        # Collect all properties except title and details for metadata
        metadata = {
            key: value for key, value in job.items()
            if key not in ['title', 'details']
        }

        # Create row with id, metadata, text, embedding
        row = {
            "id": chunk["id"],
            "metadata": metadata,
            "text": text,
            "embedding": embedding
        }
        rows_data.append(row)

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
                for chunk in chunks_with_data:
                    # Validate chunk metadata
                    required_keys = [
                        "id", "doc_id", "doc_index", "chunk_index", "num_tokens",
                        "content", "start_idx", "end_idx", "line_idx"
                    ]
                    missing_keys = [
                        key for key in required_keys if key not in chunk]
                    if missing_keys:
                        logger.error(f"Missing keys in chunk: {missing_keys}")
                        raise ValueError(
                            f"Chunk missing required keys: {missing_keys}")
                    # Find the job corresponding to chunk["doc_id"]
                    job = job_by_id.get(chunk["doc_id"])
                    if not job:
                        logger.error(
                            f"No job found for doc_id: {chunk['doc_id']}")
                        raise ValueError(
                            f"No job found for doc_id: {chunk['doc_id']}")

                    # Generate parent_id using company name
                    header = job["title"]
                    parent_header = job["company"]
                    parent_id = generate_key(job["company"])
                    header_doc_id = generate_key(job["title"])

                    cur.execute(
                        f"""
                        INSERT INTO {DEFAULT_TABLE_DATA_NAME} (
                            chunk_id,
                            doc_id,
                            doc_index,
                            chunk_index,
                            num_tokens,
                            parent_id,
                            parent_header,
                            header,
                            header_doc_id,
                            content,
                            level,
                            parent_level,
                            start_idx,
                            end_idx
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING;
                        """,
                        (
                            chunk["id"],
                            chunk["doc_id"],
                            chunk["doc_index"],
                            chunk["chunk_index"],
                            chunk["num_tokens"],
                            parent_id,  # Deterministic UUID for company
                            parent_header,  # parent_header is company
                            header_doc_id,  # header_doc_id same as parent_id
                            header,  # header is job title
                            chunk["content"],
                            1,  # level for chunk
                            0,  # parent_level for company
                            chunk["start_idx"],
                            chunk["end_idx"]
                        )
                    )
                db_client.conn.commit()
                logger.info(
                    f"Inserted {len(chunks_with_data)} metadata records into '{DEFAULT_TABLE_DATA_NAME}' table.")
        except Exception as e:
            logger.error(f"Failed to insert metadata: {str(e)}")
            db_client.conn.rollback()
            raise
        logger.success(
            f"Saved {len(rows_data)} chunked job embeddings to '{DEFAULT_TABLE_NAME}' table."
        )
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
