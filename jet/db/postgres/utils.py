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


def connect_db(dbname: str, user: str = DEFAULT_USER, password: str = DEFAULT_PASSWORD, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> "psycopg.Connection":
    """Create a new database."""
    conn = psycopg.connect(
        dbname=dbname,  # Connect to a default DB to execute drop command
        user=user,
        password=password,
        host=host,
        port=port,
        autocommit=True
    )
    return conn


def connect_default_db() -> "psycopg.Connection":
    """Create a new database."""
    return connect_db(
        dbname=DEFAULT_DB,
        user=DEFAULT_USER,
        password=DEFAULT_PASSWORD,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
    )


def create_db(dbname: str) -> None:
    """Create a new database."""
    temp_conn = connect_default_db()

    with temp_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE {dbname};")

    temp_conn.close()


def delete_db(dbname: str) -> None:
    """Delete the specified database, ensuring it is not the currently connected one."""
    temp_conn = connect_default_db()

    with temp_conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {dbname};")

    temp_conn.close()


__all__ = [
    "create_db",
    "delete_db",
]
