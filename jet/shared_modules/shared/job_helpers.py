from typing import List, Dict, Optional
import numpy as np
from jet.logger import logger
from jet.models.embeddings.base import generate_embeddings
from jet.models.model_types import EmbedModelType
from shared.data_types.job import JobData
from jet.db.postgres.pgvector import PgVectorClient, VectorInput

DEFAULT_EMBED_MODEL: EmbedModelType = "mxbai-embed-large"
DEFAULT_DB_NAME = "jobs_db1"
DEFAULT_TABLE_NAME = "embeddings"


def generate_job_embeddings(data: List[JobData], embed_model: EmbedModelType = DEFAULT_EMBED_MODEL) -> np.ndarray:
    texts = [f"{d['title']}\n{d['details']}" for d in data]
    embeddings = generate_embeddings(
        texts, embed_model, show_progress=True, return_format="numpy")
    assert isinstance(
        embeddings, np.ndarray), f"Expected np.ndarray, got {type(embeddings)}"
    return embeddings


def save_job_embeddings(jobs: List[JobData], embed_model: EmbedModelType = DEFAULT_EMBED_MODEL, db_client: Optional[PgVectorClient] = None, overwrite_db: bool = True) -> None:
    """
    Save job embeddings to the jobs_db1 embeddings table.

    Args:
        jobs: List of JobData containing job IDs
        embeddings: NumPy array of embeddings corresponding to jobs
        db_client: PgVectorClient instance for database operations
    """
    embeddings = generate_job_embeddings(jobs, embed_model)

    if len(jobs) != len(embeddings):
        raise ValueError(
            f"Mismatch between jobs ({len(jobs)}) and embeddings ({len(embeddings)})")

    vector_data: Dict[str, VectorInput] = {
        job["id"]: embedding for job, embedding in zip(jobs, embeddings)
    }

    if not db_client:
        db_client = PgVectorClient(
            dbname=DEFAULT_DB_NAME, overwrite_db=overwrite_db)

    logger.info(
        f"Inserting {len(vector_data)} job embeddings into '{DEFAULT_TABLE_NAME}' table...")
    with db_client:
        db_client.insert_vector_by_ids(DEFAULT_TABLE_NAME, vector_data)
        logger.success(
            f"Saved {len(vector_data)} job embeddings to '{DEFAULT_TABLE_NAME}' table.")
