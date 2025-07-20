import pytest
import numpy as np
from typing import List, Dict
from jet.db.postgres.pgvector import PgVectorClient
import json


@pytest.fixture
def client():
    client = PgVectorClient(overwrite_db=True)
    yield client
    client.delete_db(confirm=True)


class TestCreateRow:
    def test_create_row_with_embedding(self, client: PgVectorClient):
        # Given: A table name and row data with an embedding
        table_name = "test_table"
        embedding = np.array([1.0, 2.0, 3.0])
        row_data = {"embedding": embedding}
        expected_id = str  # Expecting a UUID string

        # When: Creating a single row
        result_id = client.create_row(table_name, row_data, dimension=3)

        # Then: The row is inserted and can be retrieved
        assert isinstance(result_id, str)
        rows = client.get_rows(table_name)
        expected_rows = [{"id": result_id, "embedding": embedding}]
        assert len(rows) == 1
        assert rows[0]["id"] == result_id
        assert np.array_equal(rows[0]["embedding"], embedding)

    def test_create_row_with_custom_id_and_metadata(self, client: PgVectorClient):
        # Given: A table with additional metadata column and custom ID
        table_name = "test_table"
        client.create_table(table_name, 3)
        client.conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN metadata TEXT;")
        custom_id = "custom-123"
        embedding = np.array([1.0, 2.0, 3.0])
        row_data = {"id": custom_id,
                    "embedding": embedding, "metadata": "test data"}
        expected_row = {"id": custom_id, "embedding": embedding}

        # When: Creating a row with custom ID and metadata
        result_id = client.create_row(table_name, row_data)

        # Then: The row is inserted with the specified ID and can be retrieved
        assert result_id == custom_id
        rows = client.get_rows(table_name)
        assert len(rows) == 1
        assert rows[0]["id"] == custom_id
        assert np.array_equal(rows[0]["embedding"], embedding)

    def test_create_row_with_additional_columns_and_nested_dict(self, client: PgVectorClient):
        # Given: A table name and row data with additional columns and nested dict
        table_name = "test_table"
        embedding = np.array([1.0, 2.0, 3.0])
        nested_data = {"key1": "value1", "key2": {"nested_key": 42}}
        row_data = {
            "embedding": embedding,
            "score": 95.5,
            "is_active": True,
            "details": nested_data
        }
        expected_id = str

        # When: Creating a row with additional columns and nested dict
        result_id = client.create_row(table_name, row_data, dimension=3)

        # Then: The row is inserted with all columns and can be queried
        assert isinstance(result_id, expected_id)
        with client.conn.cursor() as cur:
            cur.execute(
                f"SELECT embedding, score, is_active, details FROM {table_name} WHERE id = %s;",
                (result_id,)
            )
            result = cur.fetchone()
            assert np.array_equal(result["embedding"], embedding)
            assert result["score"] == 95.5
            assert result["is_active"] is True
            assert result["details"] == nested_data


class TestCreateRows:
    def test_create_rows_with_embedding_and_metadata(self, client: PgVectorClient):
        # Given: A table name and multiple rows with embeddings and metadata
        table_name = "test_table"
        rows_data = [
            {
                "embedding": np.array([1.0, 2.0, 3.0]),
                "metadata": f"row_{i}"
            } for i in range(2)
        ]
        Azeezexpected_ids = list

        # When: Creating multiple rows
        result_ids = client.create_rows(table_name, rows_data, dimension=3)

        # Then: The rows are inserted and can be retrieved
        assert isinstance(result_ids, list)
        assert len(result_ids) == 2
        rows = client.get_rows(table_name)
        expected_rows = [
            {"id": result_ids[0], "embedding": np.array([1.0, 2.0, 3.0])},
            {"id": result_ids[1], "embedding": np.array([1.0, 2.0, 3.0])}
        ]
        assert len(rows) == 2
        for row, expected in zip(rows, expected_rows):
            assert row["id"] in result_ids
            assert np.array_equal(row["embedding"], expected["embedding"])

    def test_create_rows_with_additional_columns_and_nested_dict(self, client: PgVectorClient):
        # Given: A table name and multiple rows with additional columns and nested dicts
        table_name = "test_table"
        rows_data = [
            {
                "embedding": np.array([1.0 * (i+1), 2.0 * (i+1), 3.0 * (i+1)]),
                "score": 90.0 + i,
                "is_active": i % 2 == 0,
                "details": {"index": i, "data": {"value": f"test_{i}"}}
            } for i in range(2)
        ]
        expected_ids = list

        # When: Creating multiple rows
        result_ids = client.create_rows(table_name, rows_data, dimension=3)

        # Then: The rows are inserted with all columns and can be queried
        assert isinstance(result_ids, list)
        assert len(result_ids) == 2
        with client.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding, score, is_active, details FROM {table_name} ORDER BY score;"
            )
            results = cur.fetchall()
            for i, (result, row_data) in enumerate(zip(results, rows_data)):
                assert result["id"] in result_ids
                assert np.array_equal(
                    result["embedding"], row_data["embedding"])
                assert result["score"] == row_data["score"]
                assert result["is_active"] == row_data["is_active"]
                assert result["details"] == row_data["details"]
