# jet/db/postgres/managers/metadata.py

from typing import Dict, List, Union

from jet.db.postgres.pg_types import DatabaseMetadata, TableMetadata


class MetadataManager:
    def __init__(self, conn):
        self.conn = conn

    def get_database_metadata(self) -> DatabaseMetadata:
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
                raise RuntimeError(f"Database {self.conn.info.dbname} not found")
            return {
                "dbname": result["dbname"],
                "owner": result["owner"],
                "encoding": result["encoding"],
                "collation": result["collation"],
                "ctype": result["ctype"],
                "size_mb": round(result["size_mb"], 2),
            }

    def get_all_tables(self) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            return [row["table_name"] for row in cur.fetchall()]

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        with self.conn.cursor() as cur:
            # Get table info
            cur.execute(
                "SELECT t.table_name, t.table_type, t.table_schema AS schema_name, "
                "COALESCE((SELECT reltuples::bigint FROM pg_class WHERE relname = t.table_name), 0) AS row_count "
                "FROM information_schema.tables t "
                "WHERE t.table_schema = 'public' AND t.table_name = %s;",
                (table_name,),
            )
            table_result = cur.fetchone()
            if not table_result:
                raise RuntimeError(f"Table {table_name} not found in public schema")

            # Get column info
            cur.execute(
                "SELECT column_name, data_type, is_nullable, "
                "character_maximum_length, numeric_precision, numeric_scale "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,),
            )
            columns = [dict(row) for row in cur.fetchall()]

            return {
                "table_name": table_result["table_name"],
                "table_type": table_result["table_type"],
                "schema_name": table_result["schema_name"],
                "row_count": table_result["row_count"],
                "columns": columns,
            }

    def get_database_summary(
        self,
    ) -> Dict[str, Union[DatabaseMetadata, List[TableMetadata]]]:
        tables = self.get_all_tables()
        table_metadata = [self.get_table_metadata(table) for table in tables]
        return {
            "database_metadata": self.get_database_metadata(),
            "tables": table_metadata,
        }
