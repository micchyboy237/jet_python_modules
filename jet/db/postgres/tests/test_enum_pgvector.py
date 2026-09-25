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
def cleanup_enums_and_tables(pgvector_client):
    """Automatically clean up enum types and tables after each test."""
    created_types = []
    created_tables = []

    pgvector_client._test_created_types = created_types
    pgvector_client._test_created_tables = created_tables

    yield

    conn = pgvector_client.conn

    # Cleanup: Drop tables first (they depend on enum types)
    for table_name in reversed(created_tables):
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
        except Exception as e:
            print(f"Warning: Failed to drop table {table_name}: {e}")

    # Cleanup: Drop enum types
    for type_name in reversed(created_types):
        try:
            pgvector_client.drop_enum_type(type_name)
        except Exception as e:
            print(f"Warning: Failed to drop type {type_name}: {e}")


class TestEnumWithVectorBasic:
    """Test basic enum usage with vector embeddings."""

    def test_create_enum_with_vector_table(self, pgvector_client):
        """Test creating a table with both enum and vector columns."""
        type_name = "document_status"
        pgvector_client.create_enum_type(type_name, ["draft", "published", "archived"])
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "documents",
            dimension=3,
            additional_columns={
                "title": "TEXT",
                "status": type_name,
            },
        )
        pgvector_client._test_created_tables.append("documents")

        # Verify table structure
        metadata = pgvector_client.get_table_metadata("documents")
        col_map = {c["column_name"]: c for c in metadata["columns"]}

        # Check status column uses enum
        assert col_map["status"]["data_type"] == "USER-DEFINED"

        # Check embedding column exists
        assert col_map["embedding"]["data_type"] == "USER-DEFINED"

    def test_insert_row_with_enum_and_vector(self, pgvector_client):
        """Test inserting a row with both enum value and embedding."""
        type_name = "category"
        pgvector_client.create_enum_type(type_name, ["tech", "science", "arts"])
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "articles",
            dimension=3,
            additional_columns={
                "title": "TEXT",
                "category": type_name,
            },
        )
        pgvector_client._test_created_tables.append("articles")

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
        type_name = "topic"
        pgvector_client.create_enum_type(type_name, ["python", "javascript", "rust"])
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "code_snippets",
            dimension=4,
            additional_columns={
                "language": type_name,
                "description": "TEXT",
            },
        )
        pgvector_client._test_created_tables.append("code_snippets")

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
        python_snippets = pgvector_client.get_rows(
            "code_snippets",
            where_conditions={"language": "python"},
            order_by=("id", "ASC"),
        )

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
        type_name = "priority"
        pgvector_client.create_enum_type(type_name, ["low", "medium", "high"])
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "tasks",
            dimension=2,
            additional_columns={
                "name": "TEXT",
                "priority": type_name,
            },
        )
        pgvector_client._test_created_tables.append("tasks")

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
        type_name = "status"
        pgvector_client.create_enum_type(type_name, ["todo", "in-progress", "done"])
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "projects",
            dimension=3,
            additional_columns={
                "name": "TEXT",
                "status": type_name,
            },
        )
        pgvector_client._test_created_tables.append("projects")

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
        dept_type = "department"
        seniority_type = "seniority"

        pgvector_client.create_enum_type(
            dept_type, ["engineering", "marketing", "sales"]
        )
        pgvector_client.create_enum_type(seniority_type, ["junior", "mid", "senior"])
        pgvector_client._test_created_types.extend([dept_type, seniority_type])

        pgvector_client.create_vector_table(
            "employees",
            dimension=5,
            additional_columns={
                "name": "TEXT",
                "department": dept_type,
                "seniority": seniority_type,
            },
        )
        pgvector_client._test_created_tables.append("employees")

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
        eng_seniors = pgvector_client.get_rows(
            "employees",
            where_conditions={"department": "engineering", "seniority": "senior"},
        )

        assert len(eng_seniors) == 1
        assert eng_seniors[0]["name"] == "Alice"

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
        type_name = "quality"
        pgvector_client.create_enum_type(
            type_name, ["poor", "fair", "good", "excellent"]
        )
        pgvector_client._test_created_types.append(type_name)

        pgvector_client.create_vector_table(
            "products",
            dimension=3,
            additional_columns={
                "product_name": "TEXT",
                "quality": type_name,
            },
        )
        pgvector_client._test_created_tables.append("products")

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
