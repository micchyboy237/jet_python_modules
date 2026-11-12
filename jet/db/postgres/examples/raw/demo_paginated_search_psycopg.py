"""
pgvector Pagination Demo with psycopg and embed_utils
======================================================
Demonstrates three pagination approaches for vector similarity search:
1. OFFSET/LIMIT pagination (simple but less efficient at scale)
2. Cursor-based pagination (efficient for deep pagination)
3. Batch/iterator pattern (memory-efficient streaming)

Uses jet.adapters.llama_cpp.embed_utils for embedding generation.

Requirements:
    pip install psycopg[binary] pgvector numpy rich
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import psycopg
from jet.adapters.llama_cpp.config import EMBED_DIMS, EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed, shutdown_embed_pool
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)

# Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "",  # Replace with your actual password if needed
    "host": "localhost",
    "port": 5432,
}


@dataclass
class SearchResult:
    """Represents a single search result"""

    id: int
    title: str
    content: str
    embedding: np.ndarray
    similarity: float
    created_at: datetime
    category: str


class VectorSearchWithPagination:
    """
    Demonstrates vector search with different pagination strategies.
    Uses pgvector extension for PostgreSQL to perform similarity searches
    on high-dimensional vectors with various pagination methods.

    Integrates with jet.adapters.llama_cpp.embed_utils for embedding generation.
    """

    def __init__(self):
        """Initialize database connection parameters."""
        self.conn = None
        self.vector_dim = EMBED_DIMS  # Use config value instead of hardcoded

    def connect(self):
        """Establish database connection and register vector type."""
        try:
            self.conn = psycopg.connect(**DB_CONFIG, autocommit=True)
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self.conn)
            print("✓ Connected to database and registered vector type")
        except Exception as e:
            print(f"✗ Failed to connect to database: {e}")
            raise

    def setup_schema(self):
        """Create the documents table with appropriate indexes."""
        print("\n📊 Setting up database schema...")

        # Drop existing table if it exists
        self.conn.execute("DROP TABLE IF EXISTS documents")

        # Create table with vector column using configured embedding dimensions
        self.conn.execute(f"""
            CREATE TABLE documents (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({self.vector_dim}),
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create HNSW index for efficient approximate nearest neighbor search
        # Using cosine distance (<=> operator)
        self.conn.execute("""
            CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
        """)

        # Create index on category for filtering
        self.conn.execute("CREATE INDEX idx_documents_category ON documents (category)")

        print(f"✓ Schema created with HNSW index (dim={self.vector_dim})")

    def insert_sample_data(self, num_records: int = 1000):
        """Insert sample documents with embeddings using embed_utils."""
        print(f"\n📝 Inserting {num_records} sample documents...")

        categories = ["technology", "science", "health", "business", "education"]
        titles = [
            "Introduction to Machine Learning",
            "Quantum Computing Basics",
            "Healthy Eating Habits",
            "Startup Growth Strategies",
            "Online Education Trends",
            "Artificial Intelligence Applications",
            "Climate Change Research",
            "Mental Health Awareness",
            "Digital Marketing Tips",
            "Remote Work Best Practices",
        ]

        contents = [
            "Machine learning algorithms can identify patterns in large datasets.",
            "Quantum computers leverage quantum mechanics for computation.",
            "A balanced diet includes fruits, vegetables, and whole grains.",
            "Startups need to focus on product-market fit early on.",
            "E-learning platforms have transformed education accessibility.",
            "AI is revolutionizing industries from healthcare to finance.",
            "Global warming affects ecosystems and weather patterns worldwide.",
            "Mental health support is crucial for overall well-being.",
            "Social media marketing requires consistent engagement strategies.",
            "Remote work offers flexibility but requires strong communication.",
        ]

        # Prepare all texts first
        all_texts = []
        metadata = []

        for i in range(num_records):
            idx = i % len(titles)
            category = categories[i % len(categories)]

            # Add variation to content
            content = f"{contents[idx]} This is document #{i + 1}. "
            content += f"Category: {category}. Generated for testing purposes."

            all_texts.append(content)
            metadata.append(
                {
                    "title": f"{titles[idx]} - Document {i + 1}",
                    "category": category,
                    "content": content,
                }
            )

        # Use embed_utils to generate all embeddings at once with progress bar
        print("\n🧠 Generating embeddings using llama.cpp server...")
        try:
            embeddings = embed(
                text=all_texts,
                model=EMBED_MODEL,
                return_format="list",  # Get list format for easier DB insertion
                show_progress=True,
                progress_description="Embedding documents",
            )
        except Exception as e:
            print(f"✗ Error generating embeddings: {e}")
            raise

        # Convert to list of lists if numpy array
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # Batch insert into database using COPY
        print("\n💾 Inserting into database...")
        records = []
        for i, meta in enumerate(metadata):
            # FIX: Convert embedding list to JSON string format for pgvector
            embedding_json = str(
                embeddings[i]
            )  # Converts [0.1, 0.2, ...] to "[0.1, 0.2, ...]"
            records.append(
                (meta["title"], meta["content"], embedding_json, meta["category"])
            )

            # Batch insert every 100 records
            if len(records) >= 100 or i == num_records - 1:
                with self.conn.cursor().copy(
                    "COPY documents (title, content, embedding, category) FROM STDIN"
                ) as copy:
                    for record in records:
                        copy.write_row(record)

                self.conn.commit()
                print(f"  Inserted batch of {len(records)} records via COPY")
                records = []

        print(f"✓ Inserted {num_records} documents with embeddings")

    def offset_limit_pagination(
        self,
        query_embedding: List[float],
        page_size: int = 10,
        page_number: int = 1,
        category_filter: Optional[str] = None,
    ) -> Tuple[List[SearchResult], int]:
        """
        Traditional OFFSET/LIMIT pagination for vector search.
        """
        offset = (page_number - 1) * page_size

        # Build query with optional filter
        if category_filter:
            base_query = """
                SELECT id, title, content, embedding, 
                       1 - (embedding <=> %s::vector) AS similarity,
                       created_at, category
                FROM documents
                WHERE category = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s OFFSET %s
            """
            params = (
                query_embedding,
                category_filter,
                query_embedding,
                page_size,
                offset,
            )
        else:
            base_query = """
                SELECT id, title, content, embedding, 
                       1 - (embedding <=> %s::vector) AS similarity,
                       created_at, category
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s OFFSET %s
            """
            params = (query_embedding, query_embedding, page_size, offset)

        # Get paginated results
        with self.conn.cursor() as cur:
            cur.execute(base_query, params)
            rows = cur.fetchall()

        # Get total count for pagination metadata
        if category_filter:
            count_query = "SELECT COUNT(*) FROM documents WHERE category = %s"
            total_count = self.conn.execute(count_query, (category_filter,)).fetchone()[
                0
            ]
        else:
            total_count = self.conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]

        # Convert to SearchResult objects
        results = [
            SearchResult(
                id=row[0],
                title=row[1],
                content=row[2],
                embedding=np.array(row[3]),
                similarity=float(row[4]),
                created_at=row[5],
                category=row[6],
            )
            for row in rows
        ]

        return results, total_count

    def cursor_based_pagination(
        self,
        query_embedding: List[float],
        page_size: int = 10,
        last_id: Optional[int] = None,
        last_similarity: Optional[float] = None,
        category_filter: Optional[str] = None,
    ) -> Tuple[List[SearchResult], Optional[int], Optional[float]]:
        """
        Cursor-based (keyset) pagination for efficient deep pagination.

        Uses named parameters so placeholder order can never drift out of
        sync with the params dict.
        """
        # Named params: order-independent and reusable anywhere in the SQL
        params: dict = {"emb": query_embedding, "limit": page_size}
        conditions: List[str] = []

        if last_id is not None and last_similarity is not None:
            # Parentheses keep the OR group separate from the AND category filter
            conditions.append(
                """(
                    (1 - (embedding <=> %(emb)s::vector)) < %(last_sim)s
                    OR (
                        (1 - (embedding <=> %(emb)s::vector)) = %(last_sim)s
                        AND id < %(last_id)s
                    )
                )"""
            )
            params["last_sim"] = last_similarity
            params["last_id"] = last_id

        if category_filter:
            conditions.append("category = %(category)s")
            params["category"] = category_filter

        where_sql = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT id, title, content, embedding,
                   1 - (embedding <=> %(emb)s::vector) AS similarity,
                   created_at, category
            FROM documents
            WHERE {where_sql}
            ORDER BY embedding <=> %(emb)s::vector, id DESC
            LIMIT %(limit)s
        """

        logger.info(
            "cursor page: size=%d last_id=%s last_sim=%s category=%s",
            page_size,
            last_id,
            last_similarity,
            category_filter,
        )
        logger.debug(
            "cursor params (excluding embedding): %s",
            {k: v for k, v in params.items() if k != "emb"},
        )

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results = [
            SearchResult(
                id=row[0],
                title=row[1],
                content=row[2],
                embedding=np.array(row[3]),
                similarity=float(row[4]),
                created_at=row[5],
                category=row[6],
            )
            for row in rows
        ]

        if results:
            next_id = results[-1].id
            next_similarity = results[-1].similarity
        else:
            next_id = None
            next_similarity = None

        logger.info(
            "cursor page returned %d rows, next_id=%s next_sim=%s",
            len(results),
            next_id,
            next_similarity,
        )
        return results, next_id, next_similarity

    def batch_iterator(
        self,
        query_embedding: List[float],
        batch_size: int = 50,
        max_results: Optional[int] = None,
        category_filter: Optional[str] = None,
    ):
        """
        Generator that yields batches of results for memory-efficient processing.
        """
        last_id = None
        last_similarity = None
        total_yielded = 0

        while True:
            # Get next batch
            results, next_id, next_similarity = self.cursor_based_pagination(
                query_embedding=query_embedding,
                page_size=batch_size,
                last_id=last_id,
                last_similarity=last_similarity,
                category_filter=category_filter,
            )

            # Stop if no more results
            if not results:
                break

            # Yield current batch
            yield results

            # Update counters
            total_yielded += len(results)
            last_id = next_id
            last_similarity = next_similarity

            # Check if we've reached max results
            if max_results and total_yielded >= max_results:
                break

    def demonstrate_offset_pagination(self, query_text: str):
        """Demonstrate OFFSET/LIMIT pagination."""
        print("\n" + "=" * 70)
        print("METHOD 1: OFFSET/LIMIT Pagination")
        print("=" * 70)

        # Generate query embedding using embed_utils
        print(f"\n🔍 Embedding query: '{query_text}'")
        query_embedding = embed(
            text=query_text,
            model=EMBED_MODEL,
            return_format="list",
            show_progress=False,
        )

        # Fetch multiple pages
        page_size = 5
        for page_num in range(1, 4):
            print(f"\n📄 Page {page_num} (showing {page_size} results):")
            print("-" * 70)

            results, total_count = self.offset_limit_pagination(
                query_embedding=query_embedding,
                page_size=page_size,
                page_number=page_num,
                category_filter="technology",
            )

            print(f"Total matching documents: {total_count}")
            print(
                f"Showing results {(page_num - 1) * page_size + 1}-{min(page_num * page_size, total_count)}\n"
            )

            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.similarity:.3f}] {result.title}")
                print(f"     ID: {result.id} | Category: {result.category}")

        print("\n⚠️  Note: OFFSET becomes slower for deeper pages as it scans")
        print("   and discards all preceding rows.")

    def demonstrate_cursor_pagination(self, query_text: str):
        """Demonstrate cursor-based pagination."""
        print("\n" + "=" * 70)
        print("METHOD 2: Cursor-Based (Keyset) Pagination")
        print("=" * 70)

        # Generate query embedding using embed_utils
        print(f"\n🔍 Embedding query: '{query_text}'")
        query_embedding = embed(
            text=query_text,
            model=EMBED_MODEL,
            return_format="list",
            show_progress=False,
        )

        # Fetch pages using cursor
        page_size = 5
        last_id = None
        last_similarity = None

        for page_num in range(1, 4):
            print(f"\n📄 Page {page_num} (showing {page_size} results):")
            print("-" * 70)

            results, next_id, next_similarity = self.cursor_based_pagination(
                query_embedding=query_embedding,
                page_size=page_size,
                last_id=last_id,
                last_similarity=last_similarity,
                category_filter="technology",
            )

            if not results:
                print("No more results available.")
                break

            print(
                f"Cursor for next page: id={next_id}, similarity={next_similarity:.3f}\n"
            )

            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result.similarity:.3f}] {result.title}")
                print(f"     ID: {result.id} | Category: {result.category}")

            # Update cursor for next iteration
            last_id = next_id
            last_similarity = next_similarity

        print("\n✓ Note: Cursor pagination maintains constant performance")
        print("  regardless of page depth by using indexed lookups.")

    def demonstrate_batch_processing(self, query_text: str):
        """Demonstrate batch iterator pattern."""
        print("\n" + "=" * 70)
        print("METHOD 3: Batch Iterator (Memory-Efficient Streaming)")
        print("=" * 70)

        # Generate query embedding using embed_utils
        print(f"\n🔍 Embedding query: '{query_text}'")
        query_embedding = embed(
            text=query_text,
            model=EMBED_MODEL,
            return_format="list",
            show_progress=False,
        )

        # Process results in batches
        batch_size = 10
        max_results = 30
        batch_count = 0
        total_processed = 0

        print(
            f"\nProcessing results in batches of {batch_size} (max {max_results} total):\n"
        )

        for batch in self.batch_iterator(
            query_embedding=query_embedding,
            batch_size=batch_size,
            max_results=max_results,
            category_filter="technology",
        ):
            batch_count += 1
            total_processed += len(batch)

            print(f"📦 Batch {batch_count}: {len(batch)} results")
            print("-" * 70)

            for i, result in enumerate(batch, 1):
                print(f"  {i}. [{result.similarity:.3f}] {result.title}")

            print()

        print(f"✓ Total processed: {total_processed} results in {batch_count} batches")
        print("  Memory usage remains constant regardless of total result size.")

    def cleanup(self):
        """Close database connection."""
        if self.conn:
            try:
                self.conn.close()
                print("\n✓ Database connection closed")
            except Exception as e:
                print(f"\n✗ Error closing database connection: {e}")


def main():
    """Main demonstration function."""
    print("\n" + "=" * 70)
    print("PGVECTOR PAGINATION DEMO WITH PSYCOPG & EMBED_UTILS")
    print("=" * 70)
    print(f"Using embedding model: {EMBED_MODEL}")
    print(f"Embedding dimensions: {EMBED_DIMS}")
    print(f"Database: {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

    # Initialize
    demo = VectorSearchWithPagination()

    try:
        # Connect and setup
        demo.connect()
        demo.setup_schema()
        demo.insert_sample_data(num_records=100)

        # Define test query
        query_text = "machine learning artificial intelligence"

        # Demonstrate all three pagination methods
        demo.demonstrate_offset_pagination(query_text)
        demo.demonstrate_cursor_pagination(query_text)
        demo.demonstrate_batch_processing(query_text)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("""
┌─────────────────────┬──────────────┬──────────────┬────────────────┐
│ Method              │ Performance  │ Complexity   │ Best For       │
├─────────────────────┼──────────────┼──────────────┼────────────────┤
│ OFFSET/LIMIT        │ Degrades     │ Simple       │ Small datasets │
│                     │ with depth   │              │ shallow pages  │
├─────────────────────┼──────────────┼──────────────┼────────────────┤
│ Cursor-Based        │ Constant     │ Moderate     │ Deep           │
│                     │              │              │ pagination     │
├─────────────────────┼──────────────┼──────────────┼────────────────┤
│ Batch Iterator      │ Constant     │ Moderate     │ Large result   │
│                     │              │              │ sets, streaming│
└─────────────────────┴──────────────┴──────────────┴────────────────┘

Key Takeaways:
• pgvector itself doesn't provide pagination - use standard SQL techniques
• OFFSET/LIMIT is simple but inefficient for deep pagination
• Cursor-based pagination uses indexed lookups for consistent performance
• Batch iterators combine cursor pagination with generators for memory efficiency
• Using embed_utils provides consistent embeddings across your application
        """)

    except Exception as e:
        print(f"\n✗ An error occurred during demonstration: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Always clean up resources
        demo.cleanup()

        # Shutdown embedding pool gracefully
        try:
            shutdown_embed_pool()
            print("✓ Embedding pool shutdown")
        except Exception as e:
            print(f"✗ Error shutting down embedding pool: {e}")


if __name__ == "__main__":
    main()
