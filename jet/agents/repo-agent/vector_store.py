"""
LanceDB integration for storing and searching code embeddings.
"""

from typing import Dict, List

import aiohttp
import lancedb
from config import db, models, servers
from lancedb.pydantic import LanceModel, Vector
from utils import logger


class ChunkSchema(LanceModel):
    id: str
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
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
            # Create indexes for performance
            # Note: In newer LanceDB, create_index takes the column name first
            try:
                # Use the config-based approach for newer LanceDB versions
                from lancedb.index import IvfPq

                self.table.create_index("vector", config=IvfPq(distance_type="cosine"))
            except ImportError:
                # Fallback for older versions
                self.table.create_index("vector", metric="cosine")
            except Exception as e:
                logger.warning(f"Index creation note: {e}")
        logger.info("LanceDB ready.")

    async def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            return

        logger.info(f"Embedding and storing {len(chunks)} chunks...")
        records = []

        # Batch embedding to avoid overwhelming local server
        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]
            embeddings = await self._get_embeddings(texts)

            for j, chunk in enumerate(batch):
                records.append(
                    ChunkSchema(
                        id=chunk["id"],
                        content=chunk["content"],
                        file_path=chunk["file_path"],
                        language=chunk["language"],
                        start_line=chunk["start_line"],
                        end_line=chunk["end_line"],
                        vector=embeddings[j],
                    )
                )

        self.table.add(records)
        logger.info("Storage complete.")

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            payload = {"input": texts, "model": models.EMBED_MODEL}
            async with session.post(
                f"{servers.EMBED_URL}/embeddings", json=payload
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [item["embedding"] for item in data["data"]]
                else:
                    raise Exception(f"Embedding API error: {resp.status}")

    async def search(self, query: str, k: int) -> List[Dict]:
        query_emb = await self._get_embeddings([query])
        # Ensure table is initialized before searching
        if not self.table:
            self.initialize()
        results = self.table.search(query_emb[0]).metric("cosine").limit(k).to_list()
        return results
