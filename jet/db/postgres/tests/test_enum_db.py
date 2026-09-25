"""
E2E tests for PostgreSQL ENUM type handling.

Tests creating enum types, using them in tables, and verifying data integrity.
Reuses PostgresClient for database management, enum operations, and cleanup.
"""

import uuid

import pytest
from jet.db.postgres.client import PostgresClient


def generate_test_db_name():
    """Generate a unique database name for test isolation."""
    return f"test_enum_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def postgres_client():
    """Create a PostgresClient instance with a unique test database."""
    db_name = generate_test_db_name()

    # Create client with overwrite_db=True to ensure clean state
    client = PostgresClient(dbname=db_name, overwrite_db=True)

    print(f"\n✓ Created test database: {db_name}")

    yield client

    # Cleanup: Delete the entire database
    try:
        client.delete_db(confirm=True)
        print(f"✓ Cleaned up test database: {db_name}")
    except Exception as e:
        print(f"Warning: Failed to cleanup test database {db_name}: {e}")


@pytest.fixture
def db_connection(postgres_client):
    """Provide the psycopg connection from PostgresClient."""
    yield postgres_client.conn


@pytest.fixture(autouse=True)
def cleanup_enums_and_tables(postgres_client):
    """Automatically clean up enum types and tables after each test."""
    created_types = []
    created_tables = []

    # Store references in the client for tracking
    postgres_client._test_created_types = created_types
    postgres_client._test_created_tables = created_tables

    yield

    conn = postgres_client.conn

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
            postgres_client.drop_enum_type(type_name)
        except Exception as e:
            print(f"Warning: Failed to drop type {type_name}: {e}")


class TestEnumBasicOperations:
    """Test basic enum type creation and usage."""

    def test_create_enum_type(self, postgres_client):
        """Test creating a simple enum type."""
        type_name = "test_mood"
        postgres_client.create_enum_type(type_name, ["sad", "ok", "happy"])
        postgres_client._test_created_types.append(type_name)

        # Verify the type exists
        values = postgres_client.get_enum_values(type_name)
        assert values == ["sad", "ok", "happy"]

    def test_create_table_with_enum_column(self, postgres_client):
        """Test creating a table that uses an enum type."""
        type_name = "priority_level"
        postgres_client.create_enum_type(type_name, ["low", "medium", "high"])
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "tasks",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "title": "TEXT NOT NULL",
                "priority": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("tasks")

        # Verify table structure
        metadata = postgres_client.get_table_metadata("tasks")
        priority_col = next(
            c for c in metadata["columns"] if c["column_name"] == "priority"
        )

        assert priority_col["data_type"] == "USER-DEFINED"
        # Note: udt_name isn't in standard ColumnMetadata but we can check via raw query if needed
        # For now, we trust the USER-DEFINED type indicates our enum

    def test_insert_enum_values(self, postgres_client):
        """Test inserting data with enum values."""
        type_name = "status"
        postgres_client.create_enum_type(type_name, ["pending", "active", "completed"])
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "projects",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "status": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("projects")

        # Insert rows using raw SQL for simplicity in this specific test case
        # or use create_row if we adapt it to handle SERIAL ids properly
        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (name, status) VALUES 
                ('Project A', 'pending'),
                ('Project B', 'active'),
                ('Project C', 'completed')
            """)

        rows = postgres_client.get_rows("projects", order_by=("id", "ASC"))

        assert len(rows) == 3
        assert rows[0]["name"] == "Project A"
        assert rows[0]["status"] == "pending"
        assert rows[1]["name"] == "Project B"
        assert rows[1]["status"] == "active"
        assert rows[2]["name"] == "Project C"
        assert rows[2]["status"] == "completed"

    def test_query_enum_values(self, postgres_client):
        """Test querying data using enum values in WHERE clause."""
        type_name = "color"
        postgres_client.create_enum_type(type_name, ["red", "green", "blue"])
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "items",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "color": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("items")

        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO items (name, color) VALUES 
                ('Item 1', 'red'),
                ('Item 2', 'green'),
                ('Item 3', 'blue'),
                ('Item 4', 'red')
            """)

        red_items = postgres_client.get_rows(
            "items", where_conditions={"color": "red"}, order_by=("id", "ASC")
        )

        assert len(red_items) == 2
        assert red_items[0]["name"] == "Item 1"
        assert red_items[1]["name"] == "Item 4"


class TestEnumConstraints:
    """Test enum type constraints and validation."""

    def test_invalid_enum_value_rejected(self, postgres_client):
        """Test that invalid enum values are rejected."""
        type_name = "size"
        postgres_client.create_enum_type(type_name, ["small", "medium", "large"])
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "products",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "size": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("products")

        import psycopg

        with postgres_client.conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO products (name, size) VALUES 
                    ('Product X', 'extra-large')
                """)

    def test_case_sensitive_enum_values(self, postgres_client):
        """Test that enum values are case-sensitive."""
        type_name = "level"
        postgres_client.create_enum_type(type_name, ["Basic", "Premium", "Enterprise"])
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "subscriptions",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "username": "TEXT",
                "level": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("subscriptions")

        import psycopg

        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO subscriptions (username, level) VALUES 
                ('User1', 'Basic')
            """)

        with postgres_client.conn.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO subscriptions (username, level) VALUES 
                    ('User2', 'basic')
                """)

    def test_validate_enum_value_helper(self, postgres_client):
        """Test the validate_enum_value helper method."""
        type_name = "valid_status"
        postgres_client.create_enum_type(type_name, ["open", "closed"])
        postgres_client._test_created_types.append(type_name)

        assert postgres_client.validate_enum_value(type_name, "open") is True
        assert postgres_client.validate_enum_value(type_name, "invalid") is False


class TestEnumOrdering:
    """Test enum value ordering behavior."""

    def test_enum_order_follows_declaration(self, postgres_client):
        """Test that enum ordering follows declaration order."""
        type_name = "rating"
        postgres_client.create_enum_type(
            type_name, ["poor", "fair", "good", "excellent"]
        )
        postgres_client._test_created_types.append(type_name)

        postgres_client.create_custom_table(
            "reviews",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "product": "TEXT",
                "rating": type_name,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("reviews")

        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO reviews (product, rating) VALUES 
                ('Product A', 'good'),
                ('Product B', 'poor'),
                ('Product C', 'excellent'),
                ('Product D', 'fair')
            """)

        ordered_reviews = postgres_client.get_rows(
            "reviews", order_by=("rating", "ASC")
        )

        # Updated expected order to include 'id' which is returned by get_rows
        expected_order = [
            {"id": 2, "product": "Product B", "rating": "poor"},
            {"id": 4, "product": "Product D", "rating": "fair"},
            {"id": 1, "product": "Product A", "rating": "good"},
            {"id": 3, "product": "Product C", "rating": "excellent"},
        ]

        assert ordered_reviews == expected_order

    def test_add_enum_value(self, postgres_client):
        """Test adding a new value to an existing enum type."""
        type_name = "dynamic_status"
        postgres_client.create_enum_type(type_name, ["draft"])
        postgres_client._test_created_types.append(type_name)

        # Add new value
        postgres_client.add_enum_value(type_name, "published")

        values = postgres_client.get_enum_values(type_name)
        assert values == ["draft", "published"]

    def test_rename_enum_value(self, postgres_client):
        """Test renaming a value in an existing enum type."""
        type_name = "renamable_status"
        postgres_client.create_enum_type(type_name, ["old_name"])
        postgres_client._test_created_types.append(type_name)

        # Rename value
        postgres_client.rename_enum_value(type_name, "old_name", "new_name")

        values = postgres_client.get_enum_values(type_name)
        assert values == ["new_name"]


class TestMultipleEnumTypes:
    """Test using multiple enum types in one table."""

    def test_table_with_multiple_enums(self, postgres_client):
        """Test a table with multiple enum columns."""
        dept_type = "department"
        status_type = "employment_status"

        postgres_client.create_enum_type(
            dept_type, ["engineering", "sales", "marketing"]
        )
        postgres_client.create_enum_type(
            status_type, ["full-time", "part-time", "contract"]
        )
        postgres_client._test_created_types.extend([dept_type, status_type])

        postgres_client.create_custom_table(
            "employees",
            columns={
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "department": dept_type,
                "status": status_type,
            },
            include_timestamps=False,
        )
        postgres_client._test_created_tables.append("employees")

        with postgres_client.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO employees (name, department, status) VALUES 
                ('Alice', 'engineering', 'full-time'),
                ('Bob', 'sales', 'part-time'),
                ('Charlie', 'marketing', 'contract')
            """)

        employees = postgres_client.get_rows("employees", order_by=("id", "ASC"))

        # Updated assertions to include 'id'
        assert employees[0] == {
            "id": 1,
            "name": "Alice",
            "department": "engineering",
            "status": "full-time",
        }
        assert employees[1] == {
            "id": 2,
            "name": "Bob",
            "department": "sales",
            "status": "part-time",
        }
        assert employees[2] == {
            "id": 3,
            "name": "Charlie",
            "department": "marketing",
            "status": "contract",
        }
