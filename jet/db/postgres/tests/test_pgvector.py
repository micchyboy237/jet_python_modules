import pytest
import numpy as np
from jet.db.postgres.pgvector import PgVectorClient
from typing import List


@pytest.fixture
def pgvector_client():
    client = PgVectorClient(
        dbname="vector_db1",
        user="jethroestrada",
        password="",
        host="localhost",
        port=5432
    )
    yield client
    client.delete_all_tables()
    client.close()


def test_extension_initialization(pgvector_client):
    # Given: A connected PgVectorClient
    expected_extension = "vector"
    expected_version = "0.8.0"

    # When: Checking if the vector extension is installed
    with pgvector_client.conn.cursor() as cur:
        cur.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = %s;", (expected_extension,))
        result = cur.fetchone()

    # Then: The vector extension is installed and matches expected version
    assert result is not None, "Vector extension is not installed"
    assert result["extname"] == expected_extension, f"Expected extension {expected_extension}, found {result['extname']}"
    assert result["extversion"] == expected_version, f"Expected vector version {expected_version}, found {result['extversion']}"


def test_create_and_insert_vector(pgvector_client):
    # Given: A table name and vector dimension
    table_name = "test_embeddings"
    vector_dim = 3
    expected_vector = np.random.rand(vector_dim).tolist()
    expected_id = None

    # When: Creating a table and inserting a vector
    pgvector_client.create_table(table_name, vector_dim)
    vector_id = pgvector_client.insert_vector(table_name, expected_vector)
    retrieved_vector = pgvector_client.get_vector_by_id(table_name, vector_id)

    # Then: The retrieved vector matches the inserted vector
    result = retrieved_vector
    assert result == pytest.approx(
        expected_vector, rel=1e-5), "Retrieved vector does not match inserted vector"
    assert isinstance(vector_id, str), "Vector ID should be a string"
