"""
LanceDB integration for storing and searching code embeddings.
Uses jet's production-grade embedding utilities.
"""

from typing import Dict, List

import lancedb
from config import db, models
from jet.adapters.llama_cpp.embed_utils import embed
from lancedb.pydantic import LanceModel, Vector
from utils import logger


class ChunkSchema(LanceModel):
    id: str
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    num_tokens: int  # Added for better tracking
    vector: Vector(db.DIMENSIONS)


class VectorStore:
    def __init__(self):
        self.db_conn = None
        self.table = None

    def initialize(self):
        logger.info("Initializing LanceDB...")
        self.db_conn = lancedb.connect(db.PATH)
        if db.TABLE_NAME in self.db_conn.table_names():
            self.table = self.db_conn.open_table(db.TABLE_NAME)
        else:
            self.table = self.db_conn.create_table(db.TABLE_NAME, schema=ChunkSchema)
            try:
                from lancedb.index import IvfPq

                self.table.create_index("vector", config=IvfPq(distance_type="cosine"))
            except ImportError:
                self.table.create_index("vector", metric="cosine")
            except Exception as e:
                logger.warning(f"Index creation note: {e}")
        logger.info("LanceDB ready.")

    async def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Embedding and storing {len(chunks)} chunks...")

        # Extract texts and metadata
        texts = [chunk["content"] for chunk in chunks]

        try:
            # Use jet's optimized embedding with deduplication and parallelism
            embeddings = embed(
                text=texts,
                model=models.EMBED_MODEL,
                return_format="list",
                show_progress=False,  # Set True for CLI usage
                batch_size=64,
            )

            # Ensure embeddings is a list of lists
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            # Create records
            records = []
            for i, chunk in enumerate(chunks):
                records.append(
                    ChunkSchema(
                        id=chunk["id"],
                        content=chunk["content"],
                        file_path=chunk["file_path"],
                        language=chunk["language"],
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        num_tokens=chunk.get("num_tokens", 0),
                        vector=embeddings[i],
                    )
                )

            # Add to table
            self.table.add(records)
            logger.info("Storage complete.")

        except Exception as e:
            logger.error(f"Failed to embed/store chunks: {e}")
            raise

    async def search(self, query: str, k: int) -> List[Dict]:
        if not self.table:
            self.initialize()

        try:
            # Embed query using jet's utilities
            query_embedding = embed(
                text=query,
                model=models.EMBED_MODEL,
                return_format="list",
            )

            # Convert to list if numpy array
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            # Search LanceDB
            results = (
                self.table.search(query_embedding).metric("cosine").limit(k).to_list()
            )

            logger.debug(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
