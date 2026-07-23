# jet.db.postgres.cleanup

from typing import Optional

import psycopg2
from rich.console import Console

from .config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)

console = Console()


def drop_type_if_exists(
    type_name: str,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    cascade: bool = True,
):
    """Drop a PostgreSQL type if it exists.

    Useful for cleaning up leftover types from previous runs that might
    conflict with fresh initialization.

    Args:
        type_name: Name of the PostgreSQL type to drop
        dbname: Database name (default: from config)
        user: Database user (default: from config)
        password: Database password (default: from config)
        host: Database host (default: from config)
        port: Database port (default: from config)
        cascade: Whether to use CASCADE to drop dependent objects (default: True)

    Returns:
        bool: True if type was dropped or didn't exist, False on connection error

    Example:
        >>> from jet.db.postgres.cleanup import drop_type_if_exists
        >>> drop_type_if_exists("my_custom_type")
        >>> drop_type_if_exists("another_type", cascade=False)
    """
    dbname = dbname or DEFAULT_DB
    user = user or DEFAULT_USER
    password = password or DEFAULT_PASSWORD
    host = host or DEFAULT_HOST
    port = port or DEFAULT_PORT

    cascade_clause = "CASCADE" if cascade else "RESTRICT"

    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TYPE IF EXISTS {type_name} {cascade_clause}")
            console.print(f"[dim]✓ Dropped type '{type_name}'[/dim]")
        conn.close()
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Failed to drop type '{type_name}': {e}[/yellow]")
        return False


def drop_types_if_exist(
    type_names: list[str],
    cascade: bool = True,
    **conn_kwargs,
):
    """Drop multiple PostgreSQL types if they exist.

    Args:
        type_names: List of type names to drop
        cascade: Whether to use CASCADE (default: True)
        **conn_kwargs: Optional connection parameters (dbname, user, password, host, port)

    Returns:
        list[str]: Names of types that failed to drop (empty list = all succeeded)

    Example:
        >>> from jet.db.postgres.cleanup import drop_types_if_exist
        >>> drop_types_if_exist(["type_a", "type_b"])
        >>> drop_types_if_exist(["type_c"], cascade=False, host="custom_host")
    """
    failed = []
    for type_name in type_names:
        success = drop_type_if_exists(type_name, cascade=cascade, **conn_kwargs)
        if not success:
            failed.append(type_name)

    if failed:
        console.print(f"[yellow]⚠ Failed to drop types: {', '.join(failed)}[/yellow]")
    else:
        console.print(f"[dim]✓ All {len(type_names)} type(s) processed[/dim]")

    return failed


def drop_table_if_exists(
    table_name: str,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    cascade: bool = True,
):
    """Drop a PostgreSQL table if it exists.

    Useful for cleaning up leftover tables from previous runs that might
    conflict with fresh initialization.

    Args:
        table_name: Name of the PostgreSQL table to drop
        dbname: Database name (default: from config)
        user: Database user (default: from config)
        password: Database password (default: from config)
        host: Database host (default: from config)
        port: Database port (default: from config)
        cascade: Whether to use CASCADE to drop dependent objects (default: True)

    Returns:
        bool: True if table was dropped or didn't exist, False on connection error

    Example:
        >>> from jet.db.postgres.cleanup import drop_table_if_exists
        >>> drop_table_if_exists("my_table")
        >>> drop_table_if_exists("another_table", cascade=False)
    """
    dbname = dbname or DEFAULT_DB
    user = user or DEFAULT_USER
    password = password or DEFAULT_PASSWORD
    host = host or DEFAULT_HOST
    port = port or DEFAULT_PORT

    cascade_clause = "CASCADE" if cascade else "RESTRICT"

    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name} {cascade_clause}")
            console.print(f"[dim]✓ Dropped table '{table_name}'[/dim]")
        conn.close()
        return True
    except Exception as e:
        console.print(f"[yellow]⚠ Failed to drop table '{table_name}': {e}[/yellow]")
        return False


def drop_tables_if_exist(
    table_names: list[str],
    cascade: bool = True,
    **conn_kwargs,
):
    """Drop multiple PostgreSQL tables if they exist.

    Args:
        table_names: List of table names to drop
        cascade: Whether to use CASCADE (default: True)
        **conn_kwargs: Optional connection parameters (dbname, user, password, host, port)

    Returns:
        list[str]: Names of tables that failed to drop (empty list = all succeeded)

    Example:
        >>> from jet.db.postgres.cleanup import drop_tables_if_exist
        >>> drop_tables_if_exist(["table_a", "table_b"])
        >>> drop_tables_if_exist(["table_c"], cascade=False, host="custom_host")
    """
    failed = []
    for table_name in table_names:
        success = drop_table_if_exists(table_name, cascade=cascade, **conn_kwargs)
        if not success:
            failed.append(table_name)

    if failed:
        console.print(f"[yellow]⚠ Failed to drop tables: {', '.join(failed)}[/yellow]")
    else:
        console.print(f"[dim]✓ All {len(table_names)} table(s) processed[/dim]")

    return failed
