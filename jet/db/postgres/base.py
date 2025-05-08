import psycopg
from pgvector.psycopg import register_vector
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple, TypedDict
from psycopg.rows import dict_row
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


class PostgresDB:
    """A utility class for managing PostgreSQL database operations."""

    def __init__(
        self,
        default_db: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT
    ):
        """Initialize PostgresDB with connection parameters."""
        self.default_db = default_db
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connect_db(
        self,
        dbname: str,
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None
    ) -> "psycopg.Connection":
        """Create a database connection."""
        conn = psycopg.connect(
            dbname=dbname,
            user=user or self.user,
            password=password or self.password,
            host=host or self.host,
            port=port or self.port,
            autocommit=True
        )
        return conn

    def connect_default_db(self) -> "psycopg.Connection":
        """Create a connection to the default database."""
        return self.connect_db(
            dbname=self.default_db,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )

    def create_db(self, dbname: str) -> None:
        """Create a new database."""
        with self.connect_default_db() as temp_conn:
            with temp_conn.cursor() as cur:
                cur.execute(f"CREATE DATABASE {dbname};")

    def delete_db(self, dbname: str) -> None:
        """Delete the specified database, ensuring it is not the currently connected one."""
        with self.connect_default_db() as temp_conn:
            with temp_conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {dbname};")


__all__ = ["PostgresDB"]
