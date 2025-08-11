import json
import uuid
from psycopg import connect, sql, errors
from typing import Any, List, Dict, Optional, Union
from psycopg.rows import dict_row
from jet.logger import logger
from .pg_types import (
    DatabaseMetadata,
    ColumnMetadata,
    TableRow,
    TableMetadata,
)
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


class PostgresClient:
    def __init__(
        self,
        dbname: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False
    ) -> None:
        """Ensure database exists, then connect."""
        self._ensure_database_exists(
            dbname, user, password, host, port, overwrite_db)
        self.conn = connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=True,
            row_factory=dict_row
        )

    def _ensure_database_exists(self, dbname: str, user: str, password: str, host: str, port: int, overwrite_db: bool) -> None:
        """Drop the target database if it exists and overwrite_db is True, then create a new one."""
        with connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=True
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
                exists = cur.fetchone()
                if exists and overwrite_db:
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                            "FROM pg_stat_activity "
                            "WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (dbname,)
                    )
                    cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(dbname)))
                if not exists or overwrite_db:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(dbname)))

    def _ensure_table_exists(self, table_name: str) -> None:
        """Check if table exists, create it if it doesn't."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s;",
                (table_name,)
            )
            table_exists = cur.fetchone()
            if not table_exists:
                self.create_table(table_name)

    def _ensure_columns_exist(self, table_name: str, row_data: Dict[str, Any]) -> None:
        """Ensure all columns in row_data exist in the table, adding them if necessary."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,)
            )
            existing_columns = {row["column_name"] for row in cur.fetchall()}

            for column, value in row_data.items():
                if column not in existing_columns and column != "id":
                    if isinstance(value, (dict, list)):
                        col_type = "jsonb"
                    elif isinstance(value, bool):
                        col_type = "boolean"
                    elif isinstance(value, (int, float)):
                        col_type = "numeric"
                    else:
                        col_type = "text"
                    query = sql.SQL("ALTER TABLE {} ADD COLUMN {} {};").format(
                        sql.Identifier(table_name),
                        sql.Identifier(column),
                        sql.SQL(col_type)
                    )
                    try:
                        cur.execute(query)
                        logger.success(
                            "Successfully created column %s in table %s", column, table_name)
                    except Exception as e:
                        logger.error(
                            "Failed to create column %s in table %s: %s", column, table_name, str(e))
                        raise

    def __enter__(self):
        """Begin transaction when entering 'with' block."""
        self.conn.execute("BEGIN;")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Handle commit or rollback when exiting 'with' block."""
        if self.conn is None or self.conn.closed:
            return
        if exc_type is None:
            self.conn.execute("COMMIT;")
        else:
            self.conn.execute("ROLLBACK;")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def begin_transaction(self) -> None:
        """Explicitly begin a new transaction."""
        self.conn.execute("BEGIN;")

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.execute("COMMIT;")

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.conn.execute("ROLLBACK;")

    def create_table(self, table_name: str) -> None:
        """Create a table with an ID column."""
        query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "id TEXT PRIMARY KEY"
            ");"
        ).format(sql.Identifier(table_name))
        with self.conn.cursor() as cur:
            cur.execute(query)

    def generate_unique_hash(self) -> str:
        """Generate a unique UUID v4 string."""
        return str(uuid.uuid4())

    def create_row(self, table_name: str, row_data: Dict[str, Any]) -> TableRow:
        """Insert a single row into the specified table with arbitrary column values, creating the table and columns if needed.

        Args:
            table_name: Name of the table to insert into
            row_data: Dictionary of column names and their values, including nested dicts

        Returns:
            TableRow dictionary containing the inserted row data including the generated or provided ID
        """
        if not row_data:
            raise ValueError("Cannot insert empty row data")

        self._ensure_table_exists(table_name)
        self._ensure_columns_exist(table_name, row_data)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,)
            )
            column_types = {row["column_name"]: row["data_type"]
                            for row in cur.fetchall()}

        row_id = row_data.get("id", self.generate_unique_hash())
        columns = ["id"] + [col for col in row_data.keys() if col != "id"]
        values = [row_id]
        placeholders = ["%s"]

        for col in columns[1:]:
            value = row_data[col]
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
                placeholders.append("%s::jsonb")
            else:
                values.append(value)
                placeholders.append("%s")

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *;").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(map(sql.SQL, placeholders))
        )

        with self.conn.cursor() as cur:
            try:
                cur.execute(query, values)
                result = cur.fetchone()
                return {
                    col: json.loads(result[col]) if column_types.get(col) == "jsonb" and isinstance(result[col], str) and result[col]
                    else result[col]
                    for col in result.keys()
                }
            except Exception as e:
                logger.error("Failed to insert row into %s: %s",
                             table_name, str(e))
                raise

    def create_rows(self, table_name: str, rows_data: List[Dict[str, Any]]) -> List[TableRow]:
        """Insert multiple rows into the specified table with arbitrary column values, creating the table and columns if needed.

        Args:
            table_name: Name of the table to insert into
            rows_data: List of dictionaries containing column names and their values, including nested dicts

        Returns:
            List of TableRow dictionaries containing the inserted row data including generated or provided IDs
        """
        if not rows_data:
            raise ValueError("Cannot insert empty rows data")
        self._ensure_table_exists(table_name)
        self._ensure_columns_exist(table_name, rows_data[0])
        if not all(set(row.keys()) == set(rows_data[0].keys()) for row in rows_data):
            raise ValueError("All rows must have the same columns")
        row_results = []
        for row in rows_data:
            try:
                row_result = self.create_row(table_name, row)
                row_results.append(row_result)
            except Exception as e:
                logger.error("Failed to insert row into %s: %s",
                             table_name, str(e))
                raise
        return row_results

    def update_row(self, table_name: str, row_id: str, row_data: Dict[str, Any]) -> TableRow:
        """Update a single row in the specified table with new column values, creating new columns if needed.

        Args:
            table_name: Name of the table to update
            row_id: ID of the row to update
            row_data: Dictionary of column names and their new values, including nested dicts

        Returns:
            TableRow dictionary containing the updated row data

        Raises:
            ValueError: If row_data is empty or row_id does not exist
        """
        if not row_data:
            raise ValueError("Cannot update with empty row data")

        self._ensure_table_exists(table_name)
        self._ensure_columns_exist(table_name, row_data)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,)
            )
            column_types = {row["column_name"]: row["data_type"]
                            for row in cur.fetchall()}

        columns = [col for col in row_data.keys()]
        values = []
        set_clauses = []

        for col in columns:
            value = row_data[col]
            if isinstance(value, (dict, list)):
                values.append(json.dumps(value))
                set_clauses.append(f"{col} = %s::jsonb")
            else:
                values.append(value)
                set_clauses.append(f"{col} = %s")

        values.append(row_id)
        query = sql.SQL("UPDATE {} SET {} WHERE id = %s RETURNING *;").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.SQL, set_clauses))
        )

        with self.conn.cursor() as cur:
            try:
                cur.execute(query, values)
                result = cur.fetchone()
                if not result:
                    raise ValueError(
                        f"No row found with id {row_id} in table {table_name}")
                return {
                    col: json.loads(result[col]) if column_types.get(col) == "jsonb" and isinstance(result[col], str) and result[col]
                    else result[col]
                    for col in result.keys()
                }
            except Exception as e:
                logger.error("Failed to update row %s in %s: %s",
                             row_id, table_name, str(e))
                raise

    def update_rows(self, table_name: str, rows_data: List[Dict[str, Any]]) -> List[TableRow]:
        """Update multiple rows in the specified table with new column values, creating new columns if needed.

        Args:
            table_name: Name of the table to update
            rows_data: List of dictionaries containing 'id' and column names with their new values

        Returns:
            List of TableRow dictionaries containing the updated row data

        Raises:
            ValueError: If rows_data is empty, any row lacks an 'id', or rows have inconsistent columns
        """
        if not rows_data:
            raise ValueError("Cannot update empty rows data")
        if not all("id" in row for row in rows_data):
            raise ValueError("All rows must include an 'id' field")
        self._ensure_table_exists(table_name)
        self._ensure_columns_exist(table_name, rows_data[0])
        if not all(set(row.keys()) == set(rows_data[0].keys()) for row in rows_data):
            raise ValueError("All rows must have the same columns")

        updated_rows = []
        for row in rows_data:
            try:
                updated_row = self.update_row(
                    table_name, row["id"], row)
                updated_rows.append(updated_row)
            except Exception as e:
                logger.error("Failed to update row %s in %s: %s",
                             row["id"], table_name, str(e))
                raise
        return updated_rows

    def get_row(self, table_name: str, row_id: str) -> Optional[TableRow]:
        """Retrieve a single row by ID from the specified table.

        Args:
            table_name: Name of the table to query
            row_id: ID of the row to retrieve

        Returns:
            Dictionary containing all columns for the row, or None if not found
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,)
            )
            columns = [row["column_name"] for row in cur.fetchall()]
            if not columns:
                raise ValueError(f"No columns found for table {table_name}")

            query = sql.SQL("SELECT {} FROM {} WHERE id = %s").format(
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.Identifier(table_name)
            )
            cur.execute(query, (row_id,))
            result = cur.fetchone()
            if not result:
                return None
            return {
                col: result[col] if not (col == "details" and isinstance(
                    result[col], str)) else json.loads(result[col]) if result[col] else None
                for col in columns
            }

    def get_rows(self, table_name: str, ids: Optional[List[str]] = None) -> List[TableRow]:
        """Retrieve rows from the specified table, optionally filtered by IDs.

        Args:
            table_name: Name of the table to query
            ids: Optional list of IDs to filter results

        Returns:
            List of dictionaries containing all columns for each row
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,)
            )
            column_info = {row["column_name"]: row["data_type"]
                           for row in cur.fetchall()}
            if not column_info:
                raise ValueError(f"No columns found for table {table_name}")

            columns = list(column_info.keys())
            query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.Identifier(table_name)
            )
            params = []
            if ids is not None:
                query = sql.SQL("{} WHERE id = ANY(%s)").format(query)
                params.append(ids)

            cur.execute(query, params)
            results = cur.fetchall()
            return [
                {
                    col: row[col] if not (column_info[col] == "jsonb" and isinstance(
                        row[col], str)) else json.loads(row[col]) if row[col] else None
                    for col in columns
                }
                for row in results
            ]

    def drop_all_rows(self, table_name: Optional[str] = None) -> None:
        """Delete all rows from a specific table or all tables if no table is provided."""
        with self.conn.cursor() as cur:
            if table_name:
                query = f"DELETE FROM {table_name};"
                cur.execute(query)
            else:
                cur.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
                tables = [row["tablename"] for row in cur.fetchall()]
                for table in tables:
                    cur.execute(f"DELETE FROM {table};")

    def delete_all_tables(self) -> None:
        """Drop all tables in the database."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = [row["tablename"] for row in cur.fetchall()]
            for table in tables:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                    sql.Identifier(table)))

    def delete_db(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(
                "Database deletion requires explicit confirmation by setting confirm=True")
        dbname = self.conn.info.dbname
        user = self.conn.info.user
        password = self.conn.info.password
        host = self.conn.info.host
        port = self.conn.info.port
        if not self.conn.closed:
            self.close()
        try:
            with connect(
                dbname="postgres",
                user=user,
                password=password,
                host=host,
                port=port,
                autocommit=True
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                            "FROM pg_stat_activity "
                            "WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (dbname,)
                    )
                    cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(dbname)))
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete database {dbname}: {str(e)}") from e
        self.conn = None

    def get_database_metadata(self) -> DatabaseMetadata:
        """Retrieve metadata for the current database."""
        with self.conn.cursor() as cur:
            query = (
                "SELECT d.datname AS dbname, d.datdba::regrole::text AS owner, "
                "pg_encoding_to_char(d.encoding) AS encoding, d.datcollate AS collation, "
                "d.datctype AS ctype, "
                "pg_database_size(d.datname)::float / 1024 / 1024 AS size_mb "
                "FROM pg_database d WHERE d.datname = %s;"
            )
            cur.execute(query, (self.conn.info.dbname,))
            result = cur.fetchone()
            if not result:
                raise RuntimeError(
                    f"Database {self.conn.info.dbname} not found")
            return {
                "dbname": result["dbname"],
                "owner": result["owner"],
                "encoding": result["encoding"],
                "collation": result["collation"],
                "ctype": result["ctype"],
                "size_mb": round(result["size_mb"], 2),
            }

    def get_all_tables(self) -> List[str]:
        """Retrieve a list of all table names in the public schema."""
        with self.conn.cursor() as cur:
            query = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
            cur.execute(query)
            results = cur.fetchall()
            return [row["table_name"] for row in results]

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        """Retrieve detailed metadata for a specific table."""
        with self.conn.cursor() as cur:
            table_query = (
                "SELECT t.table_name, t.table_type, t.table_schema AS schema_name, "
                "COALESCE((SELECT reltuples::bigint FROM pg_class "
                "WHERE relname = t.table_name), 0) AS row_count "
                "FROM information_schema.tables t "
                "WHERE t.table_schema = 'public' AND t.table_name = %s;"
            )
            formatted_table_query = sql.SQL(table_query).format(
                table_name=sql.Identifier(table_name))
            cur.execute(formatted_table_query, (table_name,))
            table_result = cur.fetchone()
            if not table_result:
                raise RuntimeError(
                    f"Table {table_name} not found in public schema")
            column_query = (
                "SELECT column_name, data_type, is_nullable, "
                "character_maximum_length, numeric_precision, numeric_scale "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;"
            )
            cur.execute(column_query, (table_name,))
            columns = [{
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "is_nullable": row["is_nullable"],
                "character_maximum_length": row["character_maximum_length"],
                "numeric_precision": row["numeric_precision"],
                "numeric_scale": row["numeric_scale"],
            } for row in cur.fetchall()]
            return {
                "table_name": table_result["table_name"],
                "table_type": table_result["table_type"],
                "schema_name": table_result["schema_name"],
                "row_count": table_result["row_count"],
                "columns": columns,
            }

    def get_database_summary(self) -> Dict[str, Union[DatabaseMetadata, List[TableMetadata]]]:
        """Retrieve a comprehensive summary of the database including metadata and all tables."""
        tables = self.get_all_tables()
        table_metadata = [self.get_table_metadata(table) for table in tables]
        return {
            "database_metadata": self.get_database_metadata(),
            "tables": table_metadata,
        }


__all__ = [
    "PostgresClient",
]
