import psycopg
from psycopg.rows import dict_row
from typing import Optional
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


class PostgresDB:
    """A utility class for managing PostgreSQL database connections and operations."""

    def __init__(
        self,
        default_db: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """Initialize PostgresDB with connection parameters."""
        self.default_db = default_db
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connect_db(
        self,
        dbname: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        autocommit: bool = False,
    ) -> psycopg.Connection:
        """Establish a connection to a specified database."""
        try:
            return psycopg.connect(
                dbname=dbname,
                user=user or self.user,
                password=password or self.password,
                host=host or self.host,
                port=port or self.port,
                autocommit=autocommit,
                row_factory=dict_row,
            )
        except psycopg.Error as e:
            raise Exception(
                f"Failed to connect to database {dbname}: {str(e)}")

    def connect_default_db(self) -> psycopg.Connection:
        """Connect to the default database, creating it if it doesn't exist."""
        # Check if default database exists by connecting to 'postgres'
        try:
            with self.connect_db(
                dbname="postgres",
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                autocommit=True,  # Use autocommit for database creation
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (
                            self.default_db,)
                    )
                    if not cur.fetchone():
                        cur.execute(f"CREATE DATABASE {self.default_db}")
        except psycopg.Error as e:
            raise Exception(
                f"Failed to verify or create database {self.default_db}: {str(e)}")

        # Connect to the default database
        return self.connect_db(
            dbname=self.default_db,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            autocommit=False,  # Default to transaction control for normal operations
        )

    def create_db(self, dbname: str) -> None:
        """Create a new database."""
        try:
            with self.connect_db(
                dbname="postgres",
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                autocommit=True,  # Autocommit required for CREATE DATABASE
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE DATABASE {dbname}")
        except psycopg.Error as e:
            raise Exception(f"Failed to create database {dbname}: {str(e)}")

    def delete_db(self, dbname: str) -> None:
        """Delete a specified database."""
        try:
            with self.connect_db(
                dbname="postgres",
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                autocommit=True,  # Autocommit required for DROP DATABASE
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE IF EXISTS {dbname}")
        except psycopg.Error as e:
            raise Exception(f"Failed to delete database {dbname}: {str(e)}")

    def verify_foreign_key(self, table: str, constraint: str) -> bool:
        """Verify if a foreign key constraint exists."""
        with self.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                        SELECT constraint_name
                        FROM information_schema.table_constraints
                        WHERE table_name = %s
                        AND constraint_type = 'FOREIGN KEY'
                        AND constraint_name = %s
                    """, (table, constraint))
                    return bool(cur.fetchone())
                except psycopg.Error as e:
                    raise Exception(
                        f"Failed to verify foreign key {constraint}: {str(e)}")


__all__ = ["PostgresDB"]
