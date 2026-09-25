"""
E2E tests for PostgreSQL ENUM + pgvector integration.

Tests creating enum types alongside vector embeddings in the same table,
verifying data integrity for both enum and vector columns.
Reuses PgVectorClient for database management and cleanup.
"""

import uuid

import numpy as np
import pytest
from jet.db.postgres.pgvector import PgVectorClient
from psycopg import sql


def generate_test_db_name():
    """Generate a unique database name for test isolation."""
    return f"test_enum_vec_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def pgvector_client():
    """Create a PgVectorClient instance with a unique test database."""
    db_name = generate_test_db_name()

    # Create client with overwrite_db=True to ensure clean state
    client = PgVectorClient(dbname=db_name, overwrite_db=True)

    print(f"\n✓ Created test database with pgvector: {db_name}")

    yield client

    # Cleanup: Delete the entire database
    try:
        client.delete_db(confirm=True)
        print(f"✓ Cleaned up test database: {db_name}")
    except Exception as e:
        print(f"Warning: Failed to cleanup test database {db_name}: {e}")


@pytest.fixture
def db_connection(pgvector_client):
    """Provide the psycopg connection from PgVectorClient."""
    yield pgvector_client.conn


@pytest.fixture(autouse=True)
def cleanup_enums_and_tables(db_connection):
    """Automatically clean up enum types and tables after each test."""
    created_types = []
    created_tables = []

    db_connection._test_created_types = created_types
    db_connection._test_created_tables = created_tables

    yield

    # Cleanup: Drop tables first (they depend on enum types)
    for table_name in reversed(created_tables):
        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                        sql.Identifier(table_name)
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to drop table {table_name}: {e}")

    # Cleanup: Drop enum types
    for type_name in reversed(created_types):
        try:
            with db_connection.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TYPE IF EXISTS {} CASCADE").format(
                        sql.Identifier(type_name)
                    )
                )
        except Exception as e:
            print(f"Warning: Failed to drop type {type_name}: {e}")


def create_enum_type(conn, type_name, values):
    """Helper to create an enum type and track it for cleanup."""
    with conn.cursor() as cur:
        values_sql = ", ".join([f"'{v}'" for v in values])
        query = sql.SQL("CREATE TYPE {} AS ENUM ({})").format(
            sql.Identifier(type_name), sql.SQL(values_sql)
        )
        cur.execute(query)

    conn._test_created_types.append(type_name)


def create_table_with_enum_and_vector(conn, table_name, dimension, columns):
    """Helper to create a table with enum and vector columns and track it for cleanup."""
    with conn.cursor() as cur:
        col_defs = [
            sql.SQL("id TEXT PRIMARY KEY"),
            sql.SQL("embedding vector({})").format(sql.Literal(dimension)),
            sql.SQL("created_at TIMESTAMPTZ DEFAULT NOW()"),
            sql.SQL("updated_at TIMESTAMPTZ DEFAULT NOW()"),
        ]

        for col_name, col_type in columns.items():
            col_defs.append(
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type))
            )

        columns_sql = sql.SQL(", ").join(col_defs)
        query = sql.SQL("CREATE TABLE {} ({})").format(
            sql.Identifier(table_name), columns_sql
        )
        cur.execute(query)

    conn._test_created_tables.append(table_name)


class TestEnumWithVectorBasic:
    """Test basic enum usage with vector embeddings."""

    def test_create_enum_with_vector_table(self, db_connection):
        """Test creating a table with both enum and vector columns."""
        create_enum_type(
            db_connection, "document_status", ["draft", "published", "archived"]
        )
        create_table_with_enum_and_vector(
            db_connection,
            "documents",
            dimension=3,
            columns={
                "title": "TEXT",
                "status": "document_status",
            },
        )

        # Verify table structure
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()

            col_map = {c["column_name"]: c for c in columns}

            # Check status column uses enum
            assert col_map["status"]["data_type"] == "USER-DEFINED"
            assert col_map["status"]["udt_name"] == "document_status"

            # Check embedding column exists
            assert col_map["embedding"]["data_type"] == "USER-DEFINED"
            assert col_map["embedding"]["udt_name"] == "vector"

    def test_insert_row_with_enum_and_vector(self, pgvector_client):
        """Test inserting a row with both enum value and embedding."""
        create_enum_type(pgvector_client.conn, "category", ["tech", "science", "arts"])
        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "articles",
            dimension=3,
            columns={
                "title": "TEXT",
                "category": "category",
            },
        )

        # Insert using PgVectorClient
        row_data = {
            "id": "article-001",
            "title": "AI Revolution",
            "category": "tech",
            "embedding": [0.1, 0.2, 0.3],
        }

        result = pgvector_client.create_row("articles", row_data, dimension=3)

        assert result["id"] == "article-001"
        assert result["title"] == "AI Revolution"
        assert result["category"] == "tech"
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) == 3

    def test_query_by_enum_with_vector_search(self, pgvector_client):
        """Test querying by enum value and performing vector search."""
        create_enum_type(
            pgvector_client.conn, "topic", ["python", "javascript", "rust"]
        )
        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "code_snippets",
            dimension=4,
            columns={
                "language": "topic",
                "description": "TEXT",
            },
        )

        # Insert multiple rows
        snippets = [
            {
                "id": "snippet-1",
                "language": "python",
                "description": "Hello World in Python",
                "embedding": [0.9, 0.1, 0.1, 0.1],
            },
            {
                "id": "snippet-2",
                "language": "javascript",
                "description": "Hello World in JS",
                "embedding": [0.1, 0.9, 0.1, 0.1],
            },
            {
                "id": "snippet-3",
                "language": "python",
                "description": "Python Data Science",
                "embedding": [0.85, 0.15, 0.05, 0.05],
            },
        ]

        pgvector_client.create_rows("code_snippets", snippets, dimension=4)

        # Query by enum value
        with pgvector_client.conn.cursor() as cur:
            cur.execute("""
                SELECT id, language, description 
                FROM code_snippets 
                WHERE language = 'python'
                ORDER BY id
            """)
            python_snippets = cur.fetchall()

            assert len(python_snippets) == 2
            assert python_snippets[0]["id"] == "snippet-1"
            assert python_snippets[1]["id"] == "snippet-3"

        # Vector search within enum category
        query_embedding = [0.88, 0.12, 0.08, 0.08]
        results = pgvector_client.search("code_snippets", query_embedding, top_k=2)

        assert len(results) == 2
        # Results should be ordered by similarity
        assert results[0]["rank"] == 1
        assert results[1]["rank"] == 2


class TestEnumConstraintsWithVectors:
    """Test enum constraints work correctly with vector operations."""

    def test_invalid_enum_rejected_with_vector(self, pgvector_client):
        """Test that invalid enum values are rejected even with valid vectors."""
        create_enum_type(pgvector_client.conn, "priority", ["low", "medium", "high"])
        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "tasks",
            dimension=2,
            columns={
                "name": "TEXT",
                "priority": "priority",
            },
        )

        # Try to insert with invalid enum but valid vector
        row_data = {
            "id": "task-001",
            "name": "Important Task",
            "priority": "urgent",  # Invalid!
            "embedding": [0.5, 0.5],
        }

        with pytest.raises(Exception):  # Could be InvalidTextRepresentation or other
            pgvector_client.create_row("tasks", row_data, dimension=2)

    def test_update_enum_value_with_vector(self, pgvector_client):
        """Test updating enum value while keeping vector intact."""
        create_enum_type(
            pgvector_client.conn, "status", ["todo", "in-progress", "done"]
        )
        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "projects",
            dimension=3,
            columns={
                "name": "TEXT",
                "status": "status",
            },
        )

        # Insert initial row
        row_data = {
            "id": "proj-001",
            "name": "Website Redesign",
            "status": "todo",
            "embedding": [0.3, 0.3, 0.4],
        }
        pgvector_client.create_row("projects", row_data, dimension=3)

        # Update enum value
        update_data = {
            "id": "proj-001",
            "status": "in-progress",
        }
        updated = pgvector_client.update_row(
            "projects", "proj-001", update_data, dimension=3
        )

        assert updated["status"] == "in-progress"

        # Verify vector is unchanged (use np.allclose for array comparison)
        retrieved = pgvector_client.get_row("projects", "proj-001")
        assert np.allclose(retrieved["embedding"], [0.3, 0.3, 0.4])


class TestMultipleEnumsWithVectors:
    """Test multiple enum types with vector embeddings."""

    def test_table_with_multiple_enums_and_vector(self, pgvector_client):
        """Test a table with multiple enum columns and vector."""
        create_enum_type(
            pgvector_client.conn, "department", ["engineering", "marketing", "sales"]
        )
        create_enum_type(pgvector_client.conn, "seniority", ["junior", "mid", "senior"])

        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "employees",
            dimension=5,
            columns={
                "name": "TEXT",
                "department": "department",
                "seniority": "seniority",
            },
        )

        # Insert employee data with embeddings
        employees = [
            {
                "id": "emp-001",
                "name": "Alice",
                "department": "engineering",
                "seniority": "senior",
                "embedding": [0.8, 0.7, 0.6, 0.5, 0.4],
            },
            {
                "id": "emp-002",
                "name": "Bob",
                "department": "marketing",
                "seniority": "mid",
                "embedding": [0.4, 0.5, 0.6, 0.7, 0.8],
            },
            {
                "id": "emp-003",
                "name": "Charlie",
                "department": "engineering",
                "seniority": "junior",
                "embedding": [0.2, 0.3, 0.4, 0.5, 0.6],
            },
        ]

        pgvector_client.create_rows("employees", employees, dimension=5)

        # Query by multiple enum filters
        with pgvector_client.conn.cursor() as cur:
            cur.execute("""
                SELECT name, department, seniority 
                FROM employees 
                WHERE department = 'engineering' AND seniority = 'senior'
            """)
            result = cur.fetchone()

            assert result is not None
            assert result["name"] == "Alice"
            assert result["department"] == "engineering"
            assert result["seniority"] == "senior"

        # Vector search filtered by enum
        query_embedding = [0.75, 0.65, 0.55, 0.45, 0.35]
        all_results = pgvector_client.search("employees", query_embedding, top_k=3)

        # Filter results by department
        engineering_results = [
            r for r in all_results if r.get("department") == "engineering"
        ]
        assert len(engineering_results) == 2  # Alice and Charlie


class TestEnumOrderingWithVectorSearch:
    """Test that enum ordering doesn't interfere with vector operations."""

    def test_vector_search_independent_of_enum_order(self, pgvector_client):
        """Test that vector search works correctly regardless of enum declaration order."""
        create_enum_type(
            pgvector_client.conn, "quality", ["poor", "fair", "good", "excellent"]
        )
        create_table_with_enum_and_vector(
            pgvector_client.conn,
            "products",
            dimension=3,
            columns={
                "product_name": "TEXT",
                "quality": "quality",
            },
        )

        # Insert products with varying quality and embeddings
        products = [
            {
                "id": "p1",
                "product_name": "Widget A",
                "quality": "poor",
                "embedding": [0.1, 0.1, 0.1],
            },
            {
                "id": "p2",
                "product_name": "Widget B",
                "quality": "excellent",
                "embedding": [0.9, 0.9, 0.9],
            },
            {
                "id": "p3",
                "product_name": "Widget C",
                "quality": "good",
                "embedding": [0.7, 0.7, 0.7],
            },
        ]

        pgvector_client.create_rows("products", products, dimension=3)

        # Search should be based on vector similarity, not enum order
        query_embedding = [0.85, 0.85, 0.85]
        results = pgvector_client.search("products", query_embedding, top_k=2)

        # Widget B should be closest (highest similarity)
        assert results[0]["id"] == "p2"
        assert results[0]["quality"] == "excellent"

        # Widget C should be second closest
        assert results[1]["id"] == "p3"
        assert results[1]["quality"] == "good"
