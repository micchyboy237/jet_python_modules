# jet/db/postgres/managers/query.py

import json
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from psycopg import sql

from jet.transformers.object import make_serializable


class QueryExecutor:
    def __init__(self, conn):
        self.conn = conn

    def generate_unique_hash(self) -> str:
        return str(uuid.uuid4())

    def _parse_row(self, row: Dict, column_types: Dict[str, str]) -> Dict:
        parsed = {}
        for col, val in row.items():
            if val is None:
                parsed[col] = None
            elif col == "embedding":
                # Handle various vector return types from pgvector
                if isinstance(val, np.ndarray):
                    parsed[col] = val.tolist()
                elif isinstance(val, list):
                    parsed[col] = val
                elif isinstance(val, str):
                    # Try to parse string representation of vector e.g. "[0.1,0.2]"
                    try:
                        parsed[col] = json.loads(val)
                    except json.JSONDecodeError:
                        parsed[col] = val
                else:
                    parsed[col] = val
            elif column_types.get(col) == "jsonb" and isinstance(val, str):
                try:
                    parsed[col] = json.loads(val) if val else None
                except json.JSONDecodeError:
                    parsed[col] = val
            else:
                parsed[col] = val
        return parsed

    def insert_row(
        self,
        table_name: str,
        data: Dict[str, Any],
        type_casts: Optional[Dict[str, str]] = None,
    ) -> Dict:
        if not data:
            raise ValueError("Cannot insert empty row")

        if "id" not in data:
            data["id"] = self.generate_unique_hash()

        columns = list(data.keys())
        values = []
        placeholders = []

        for col in columns:
            val = data[col]
            cast = type_casts.get(col) if type_casts else None

            if cast:
                values.append(val)
                placeholders.append(f"%s{cast}")
            elif isinstance(val, (dict, list)):
                values.append(json.dumps(make_serializable(val)))
                placeholders.append("%s::jsonb")
            else:
                values.append(val)
                placeholders.append("%s")

        query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) RETURNING *;").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(map(sql.SQL, placeholders)),
        )

        with self.conn.cursor() as cur:
            cur.execute(query, values)
            result = cur.fetchone()

            # Get column types for parsing
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,),
            )
            col_types = {r["column_name"]: r["data_type"] for r in cur.fetchall()}

            return self._parse_row(result, col_types)

    def update_row(
        self,
        table_name: str,
        row_id: str,
        data: Dict[str, Any],
        type_casts: Optional[Dict[str, str]] = None,
    ) -> Dict:
        if not data:
            raise ValueError("Cannot update with empty data")

        set_clauses = []
        values = []

        for col, val in data.items():
            cast = type_casts.get(col) if type_casts else None
            if cast:
                set_clauses.append(f"{col} = %s{cast}")
                values.append(val)
            elif isinstance(val, (dict, list)):
                set_clauses.append(f"{col} = %s::jsonb")
                values.append(json.dumps(make_serializable(val)))
            else:
                set_clauses.append(f"{col} = %s")
                values.append(val)

        values.append(row_id)
        query = sql.SQL(
            "UPDATE {} SET updated_at = NOW(), {} WHERE id = %s RETURNING *;"
        ).format(
            sql.Identifier(table_name), sql.SQL(", ").join(map(sql.SQL, set_clauses))
        )

        with self.conn.cursor() as cur:
            cur.execute(query, values)
            result = cur.fetchone()
            if not result:
                raise ValueError(f"Row {row_id} not found in {table_name}")

            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,),
            )
            col_types = {r["column_name"]: r["data_type"] for r in cur.fetchall()}

            return self._parse_row(result, col_types)

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
    ) -> List[Dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s;",
                (table_name,),
            )
            column_info = {
                row["column_name"]: row["data_type"] for row in cur.fetchall()
            }
            if not column_info:
                return []

            columns = list(column_info.keys())
            formatted_columns = []
            select_columns = []

            if group_by:
                for col in group_by:
                    formatted_columns.append(sql.SQL("{}").format(sql.Identifier(col)))
                    select_columns.append(col)

            if aggregates:
                for alias, (col, agg_func) in aggregates.items():
                    formatted_columns.append(
                        sql.SQL("{}({}) AS {}").format(
                            sql.SQL(agg_func),
                            sql.Identifier(col),
                            sql.Identifier(alias),
                        )
                    )
                    select_columns.append(alias)
            else:
                if not group_by:
                    formatted_columns = [
                        sql.SQL(
                            "TO_CHAR({}, 'YYYY-MM-DD\"T\"HH24:MI:SS.MS') AS {}"
                        ).format(sql.Identifier(c), sql.Identifier(c))
                        if column_info[c] == "timestamp with time zone"
                        else sql.Identifier(c)
                        for c in columns
                    ]
                    select_columns = columns

            query = sql.SQL("SELECT {} FROM {}").format(
                sql.SQL(", ").join(formatted_columns), sql.Identifier(table_name)
            )
            params = []
            where_clauses = []

            if ids is not None:
                where_clauses.append(
                    sql.SQL("{} = ANY(%s)").format(sql.Identifier(id_column))
                )
                params.append(ids)
            if where_conditions:
                for k, v in where_conditions.items():
                    where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
                    params.append(v)

            if where_clauses:
                query = sql.SQL("{} WHERE {}").format(
                    query, sql.SQL(" AND ").join(where_clauses)
                )
            if group_by:
                query = sql.SQL("{} GROUP BY {}").format(
                    query, sql.SQL(", ").join(map(sql.Identifier, group_by))
                )
            if order_by:
                query = sql.SQL("{} ORDER BY {} {}").format(
                    query, sql.Identifier(order_by[0]), sql.SQL(order_by[1])
                )
            if limit:
                query = sql.SQL("{} LIMIT %s").format(query)
                params.append(limit)

            cur.execute(query, params)
            results = cur.fetchall()

            return [self._parse_row(r, column_info) for r in results]

    def execute_raw_query(
        self, query: str, params: tuple | list = (), fetch_all: bool = True
    ):
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall() if fetch_all else cur.fetchone()

    def drop_all_rows(self, table_name: Optional[str] = None):
        with self.conn.cursor() as cur:
            if table_name:
                cur.execute(f"DELETE FROM {table_name};")
            else:
                cur.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
                )
                tables = [row["tablename"] for row in cur.fetchall()]
                for table in tables:
                    cur.execute(f"DELETE FROM {table};")
