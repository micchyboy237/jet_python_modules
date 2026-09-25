"""
E2E tests for PostgreSQL ENUM type handling.

Tests creating enum types, using them in tables, and verifying data integrity.
Reuses PostgresClient for database management and cleanup.
"""

import uuid

import psycopg
import pytest
from jet.db.postgres.client import PostgresClient
from psycopg import sql


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


def create_table_with_enum(conn, table_name, columns):
    """Helper to create a table with enum columns and track it for cleanup."""
    with conn.cursor() as cur:
        col_defs = []
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


class TestEnumBasicOperations:
    """Test basic enum type creation and usage."""

    def test_create_enum_type(self, db_connection):
        """Test creating a simple enum type."""
        create_enum_type(db_connection, "test_mood", ["sad", "ok", "happy"])

        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT typname FROM pg_type 
                WHERE typname = 'test_mood' AND typtype = 'e'
            """)
            result = cur.fetchone()
            assert result is not None
            assert result["typname"] == "test_mood"

    def test_create_table_with_enum_column(self, db_connection):
        """Test creating a table that uses an enum type."""
        create_enum_type(db_connection, "priority_level", ["low", "medium", "high"])
        create_table_with_enum(
            db_connection,
            "tasks",
            {
                "id": "SERIAL PRIMARY KEY",
                "title": "TEXT NOT NULL",
                "priority": "priority_level",
            },
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type, udt_name 
                FROM information_schema.columns 
                WHERE table_name = 'tasks'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()

            priority_col = [c for c in columns if c["column_name"] == "priority"][0]
            assert priority_col["data_type"] == "USER-DEFINED"
            assert priority_col["udt_name"] == "priority_level"

    def test_insert_enum_values(self, db_connection):
        """Test inserting data with enum values."""
        create_enum_type(db_connection, "status", ["pending", "active", "completed"])
        create_table_with_enum(
            db_connection,
            "projects",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "status": "status"},
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO projects (name, status) VALUES 
                ('Project A', 'pending'),
                ('Project B', 'active'),
                ('Project C', 'completed')
            """)

        with db_connection.cursor() as cur:
            cur.execute("SELECT name, status FROM projects ORDER BY id")
            rows = cur.fetchall()

            assert len(rows) == 3
            assert rows[0] == {"name": "Project A", "status": "pending"}
            assert rows[1] == {"name": "Project B", "status": "active"}
            assert rows[2] == {"name": "Project C", "status": "completed"}

    def test_query_enum_values(self, db_connection):
        """Test querying data using enum values in WHERE clause."""
        create_enum_type(db_connection, "color", ["red", "green", "blue"])
        create_table_with_enum(
            db_connection,
            "items",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "color": "color"},
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO items (name, color) VALUES 
                ('Item 1', 'red'),
                ('Item 2', 'green'),
                ('Item 3', 'blue'),
                ('Item 4', 'red')
            """)

        with db_connection.cursor() as cur:
            cur.execute("SELECT name FROM items WHERE color = 'red' ORDER BY id")
            red_items = cur.fetchall()

            assert len(red_items) == 2
            assert red_items[0]["name"] == "Item 1"
            assert red_items[1]["name"] == "Item 4"


class TestEnumConstraints:
    """Test enum type constraints and validation."""

    def test_invalid_enum_value_rejected(self, db_connection):
        """Test that invalid enum values are rejected."""
        create_enum_type(db_connection, "size", ["small", "medium", "large"])
        create_table_with_enum(
            db_connection,
            "products",
            {"id": "SERIAL PRIMARY KEY", "name": "TEXT", "size": "size"},
        )

        with db_connection.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO products (name, size) VALUES 
                    ('Product X', 'extra-large')
                """)

    def test_case_sensitive_enum_values(self, db_connection):
        """Test that enum values are case-sensitive."""
        create_enum_type(db_connection, "level", ["Basic", "Premium", "Enterprise"])
        create_table_with_enum(
            db_connection,
            "subscriptions",
            {"id": "SERIAL PRIMARY KEY", "username": "TEXT", "level": "level"},
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO subscriptions (username, level) VALUES 
                ('User1', 'Basic')
            """)

        with db_connection.cursor() as cur:
            with pytest.raises(psycopg.errors.InvalidTextRepresentation):
                cur.execute("""
                    INSERT INTO subscriptions (username, level) VALUES 
                    ('User2', 'basic')
                """)


class TestEnumOrdering:
    """Test enum value ordering behavior."""

    def test_enum_order_follows_declaration(self, db_connection):
        """Test that enum ordering follows declaration order."""
        create_enum_type(db_connection, "rating", ["poor", "fair", "good", "excellent"])
        create_table_with_enum(
            db_connection,
            "reviews",
            {"id": "SERIAL PRIMARY KEY", "product": "TEXT", "rating": "rating"},
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO reviews (product, rating) VALUES 
                ('Product A', 'good'),
                ('Product B', 'poor'),
                ('Product C', 'excellent'),
                ('Product D', 'fair')
            """)

        with db_connection.cursor() as cur:
            cur.execute("SELECT product, rating FROM reviews ORDER BY rating")
            ordered_reviews = cur.fetchall()

            expected_order = [
                {"product": "Product B", "rating": "poor"},
                {"product": "Product D", "rating": "fair"},
                {"product": "Product A", "rating": "good"},
                {"product": "Product C", "rating": "excellent"},
            ]

            assert ordered_reviews == expected_order


class TestMultipleEnumTypes:
    """Test using multiple enum types in one table."""

    def test_table_with_multiple_enums(self, db_connection):
        """Test a table with multiple enum columns."""
        create_enum_type(
            db_connection, "department", ["engineering", "sales", "marketing"]
        )
        create_enum_type(
            db_connection, "employment_status", ["full-time", "part-time", "contract"]
        )

        create_table_with_enum(
            db_connection,
            "employees",
            {
                "id": "SERIAL PRIMARY KEY",
                "name": "TEXT",
                "department": "department",
                "status": "employment_status",
            },
        )

        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO employees (name, department, status) VALUES 
                ('Alice', 'engineering', 'full-time'),
                ('Bob', 'sales', 'part-time'),
                ('Charlie', 'marketing', 'contract')
            """)

        with db_connection.cursor() as cur:
            cur.execute("SELECT name, department, status FROM employees ORDER BY id")
            employees = cur.fetchall()

            assert employees[0] == {
                "name": "Alice",
                "department": "engineering",
                "status": "full-time",
            }
            assert employees[1] == {
                "name": "Bob",
                "department": "sales",
                "status": "part-time",
            }
            assert employees[2] == {
                "name": "Charlie",
                "department": "marketing",
                "status": "contract",
            }
