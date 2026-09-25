# jet/db/postgres/managers/schema.py

from typing import Any, Dict, List, Optional

from psycopg import sql

from jet.logger import logger


class SchemaManager:
    def __init__(self, conn):
        self.conn = conn

    def create_enum_type(self, type_name: str, values: List[str]):
        values_sql = ", ".join([f"'{v}'" for v in values])
        query = sql.SQL("CREATE TYPE {} AS ENUM ({})").format(
            sql.Identifier(type_name), sql.SQL(values_sql)
        )
        with self.conn.cursor() as cur:
            cur.execute(query)
        logger.success("Created enum type: %s", type_name)

    def drop_enum_type(self, type_name: str, cascade: bool = True):
        cascade_clause = " CASCADE" if cascade else ""
        query = sql.SQL("DROP TYPE IF EXISTS {}{}").format(
            sql.Identifier(type_name), sql.SQL(cascade_clause)
        )
        with self.conn.cursor() as cur:
            cur.execute(query)

    def get_enum_values(self, type_name: str) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT e.enumlabel FROM pg_type t JOIN pg_enum e ON t.oid = e.enumtypid WHERE t.typname = %s ORDER BY e.enumsortorder",
                (type_name,),
            )
            return [row["enumlabel"] for row in cur.fetchall()]

    def validate_enum_value(self, type_name: str, value: str) -> bool:
        return value in self.get_enum_values(type_name)

    def add_enum_value(
        self,
        type_name: str,
        new_value: str,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ):
        with self.conn.cursor() as cur:
            if before:
                cur.execute(
                    sql.SQL("ALTER TYPE {} ADD VALUE {} BEFORE {}").format(
                        sql.Identifier(type_name),
                        sql.Literal(new_value),
                        sql.Literal(before),
                    )
                )
            elif after:
                cur.execute(
                    sql.SQL("ALTER TYPE {} ADD VALUE {} AFTER {}").format(
                        sql.Identifier(type_name),
                        sql.Literal(new_value),
                        sql.Literal(after),
                    )
                )
            else:
                cur.execute(
                    sql.SQL("ALTER TYPE {} ADD VALUE {}").format(
                        sql.Identifier(type_name), sql.Literal(new_value)
                    )
                )

    def rename_enum_value(self, type_name: str, old_value: str, new_value: str):
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL("ALTER TYPE {} RENAME VALUE {} TO {}").format(
                    sql.Identifier(type_name),
                    sql.Literal(old_value),
                    sql.Literal(new_value),
                )
            )

    def create_table(
        self, table_name: str, columns: Dict[str, str], include_timestamps: bool = True
    ):
        col_defs = []
        for col_name, col_type in columns.items():
            col_defs.append(
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type))
            )

        if include_timestamps:
            col_defs.append(sql.SQL("created_at TIMESTAMPTZ DEFAULT NOW()"))
            col_defs.append(sql.SQL("updated_at TIMESTAMPTZ DEFAULT NOW()"))

        query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name), sql.SQL(", ").join(col_defs)
        )

        with self.conn.cursor() as cur:
            cur.execute(query)
        logger.success("Created table: %s", table_name)

    def create_custom_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        primary_key: str = "id",
        include_timestamps: bool = True,
    ):
        col_defs = []
        if primary_key in columns:
            col_type = columns[primary_key].replace(" PRIMARY KEY", "").strip()
            col_defs.append(
                sql.SQL("{} {} PRIMARY KEY").format(
                    sql.Identifier(primary_key), sql.SQL(col_type)
                )
            )
        else:
            col_defs.append(sql.SQL("id TEXT PRIMARY KEY"))

        for col_name, col_type in columns.items():
            if col_name == primary_key:
                continue
            col_defs.append(
                sql.SQL("{} {}").format(sql.Identifier(col_name), sql.SQL(col_type))
            )

        if include_timestamps:
            col_defs.append(sql.SQL("created_at TIMESTAMPTZ DEFAULT NOW()"))
            col_defs.append(sql.SQL("updated_at TIMESTAMPTZ DEFAULT NOW()"))

        query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
            sql.Identifier(table_name), sql.SQL(", ").join(col_defs)
        )

        with self.conn.cursor() as cur:
            cur.execute(query)
        logger.success("Created custom table: %s", table_name)

    def ensure_columns_exist(self, table_name: str, row_data: Dict[str, Any]):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,),
            )
            existing_columns = {row["column_name"] for row in cur.fetchall()}

            for column, value in row_data.items():
                if column in existing_columns or column in {
                    "id",
                    "created_at",
                    "updated_at",
                }:
                    continue

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
                    sql.SQL(col_type),
                )
                cur.execute(query)
                logger.success("Added column %s to %s", column, table_name)

    def delete_all_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = [row["tablename"] for row in cur.fetchall()]
            for table in tables:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                        sql.Identifier(table)
                    )
                )
