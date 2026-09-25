from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from .managers.connection import ConnectionManager
from .managers.metadata import MetadataManager
from .managers.query import QueryExecutor
from .managers.schema import SchemaManager
from .pg_types import DatabaseMetadata, TableMetadata, TableRow


class PostgresClient:
    def __init__(
        self,
        dbname: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
    ):
        self.connection = ConnectionManager(
            dbname, user, password, host, port, overwrite_db
        )
        self.schema = SchemaManager(self.connection.conn)
        self.query = QueryExecutor(self.connection.conn)
        self.metadata = MetadataManager(self.connection.conn)

        # Backward compatibility
        self.conn = self.connection.conn

    # --- Schema & Enum Wrappers ---
    def create_table(self, table_name: str):
        self.schema.create_table(table_name, {"id": "TEXT PRIMARY KEY"})

    def create_custom_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        primary_key: str = "id",
        include_timestamps: bool = True,
    ):
        self.schema.create_custom_table(
            table_name, columns, primary_key, include_timestamps
        )

    def create_enum_type(self, type_name: str, values: List[str]):
        self.schema.create_enum_type(type_name, values)

    def drop_enum_type(self, type_name: str, cascade: bool = True):
        self.schema.drop_enum_type(type_name, cascade)

    def get_enum_values(self, type_name: str) -> List[str]:
        return self.schema.get_enum_values(type_name)

    def add_enum_value(
        self, type_name: str, new_value: str, before: str = None, after: str = None
    ):
        self.schema.add_enum_value(type_name, new_value, before, after)

    def rename_enum_value(self, type_name: str, old_value: str, new_value: str):
        self.schema.rename_enum_value(type_name, old_value, new_value)

    def validate_enum_value(self, type_name: str, value: str) -> bool:
        return self.schema.validate_enum_value(type_name, value)

    # --- CRUD Wrappers ---
    def create_row(self, table_name: str, row_data: Dict[str, Any]) -> TableRow:
        self.schema.ensure_columns_exist(table_name, row_data)
        return self.query.insert_row(table_name, row_data)

    def create_rows(
        self, table_name: str, rows_data: List[Dict[str, Any]]
    ) -> List[TableRow]:
        return [self.create_row(table_name, r) for r in rows_data]

    def update_row(
        self, table_name: str, row_id: str, row_data: Dict[str, Any]
    ) -> TableRow:
        return self.query.update_row(table_name, row_id, row_data)

    def update_rows(
        self, table_name: str, rows_data: List[Dict[str, Any]]
    ) -> List[TableRow]:
        return [self.update_row(table_name, r["id"], r) for r in rows_data]

    def create_or_update_row(
        self, table_name: str, row_data: Dict[str, Any]
    ) -> TableRow:
        self.schema.ensure_columns_exist(table_name, row_data)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM {} WHERE id = %s".format(table_name), (row_data["id"],)
            )
            if cur.fetchone():
                return self.update_row(table_name, row_data["id"], row_data)
            else:
                return self.create_row(table_name, row_data)

    def create_or_update_rows(
        self, table_name: str, rows_data: List[Dict[str, Any]]
    ) -> List[TableRow]:
        return [self.create_or_update_row(table_name, r) for r in rows_data]

    def get_row(self, table_name: str, row_id: str) -> Optional[TableRow]:
        rows = self.query.get_rows(table_name, where_conditions={"id": row_id})
        return rows[0] if rows else None

    def get_rows(
        self,
        table_name: str,
        ids: Optional[List[str]] = None,
        id_column: str = "id",
        where_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        order_by: Optional[Tuple[str, Literal["ASC", "DESC"]]] = None,
        group_by: Optional[List[str]] = None,
        aggregates: Optional[
            Dict[str, Tuple[str, Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]]]
        ] = None,
    ) -> List[TableRow]:
        return self.query.get_rows(
            table_name,
            ids,
            id_column,
            where_conditions,
            limit,
            order_by,
            group_by,
            aggregates,
        )

    # --- Metadata & Cleanup Wrappers ---
    def get_database_metadata(self) -> DatabaseMetadata:
        return self.metadata.get_database_metadata()

    def get_all_tables(self) -> List[str]:
        return self.metadata.get_all_tables()

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        return self.metadata.get_table_metadata(table_name)

    def get_database_summary(
        self,
    ) -> Dict[str, Union[DatabaseMetadata, List[TableMetadata]]]:
        return self.metadata.get_database_summary()

    def execute_raw_query(
        self, query: str, params: tuple | list = (), fetch_all: bool = True
    ):
        return self.query.execute_raw_query(query, params, fetch_all)

    def drop_all_rows(self, table_name: Optional[str] = None):
        self.query.drop_all_rows(table_name)

    def delete_all_tables(self):
        self.schema.delete_all_tables()

    def generate_unique_hash(self) -> str:
        return self.query.generate_unique_hash()

    def delete_db(self, confirm: bool = False):
        if confirm:
            self.connection.delete_db()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self.connection.__enter__()

    def __exit__(self, *args):
        return self.connection.__exit__(*args)


__all__ = ["PostgresClient"]
