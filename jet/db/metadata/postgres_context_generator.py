"""
PostgreSQL Context Generator for RAG (Retrieval-Augmented Generation)

Generates comprehensive database schema, metadata, sample data, and query guidance
from PostgreSQL databases. The output is formatted as markdown text suitable for
inclusion in LLM prompts to provide context about database structure and content.

Usage Examples:
    # Generate context for all tables in public schema
    from jet.db.metadata.postgres_context_generator import generate_rag_context
    context = generate_rag_context("postgresql://user:pass@localhost:5432/db")

    # Generate context for specific tables only (comma-separated)
    context = generate_rag_context(
        "postgresql://user:pass@localhost:5432/db",
        tables_filter=["users", "orders", "products"]
    )

    # With custom settings
    from jet.db.metadata.postgres_context_generator import ContextSettings
    settings = ContextSettings(sample_rows=5, include_stats=False)
    context = generate_rag_context(
        "postgresql://user:pass@localhost:5432/db",
        tables_filter=["jobs", "entities"],
        settings=settings
    )

    # Command line usage:
    # python postgres_context_generator.py -t jobs,entities -r 5 -c 80
    # python postgres_context_generator.py --tables users,orders,products
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

SIMPLE_CATEGORIES = {"N", "D", "B"}
BINARY_TYPES = {"bytea"}


@dataclass
class ContextSettings:
    """All tunable options in one place. Defaults match the original behavior."""

    sample_rows: int = 3
    sample_chars: int = 50
    header_chars: int = 60
    column_sample_chars: Dict[str, int] = field(default_factory=dict)
    exclude_sample_columns: Set[str] = field(default_factory=set)
    max_stats_tables: Optional[int] = 5
    statement_timeout_ms: int = 30000
    include_foreign_keys: bool = True
    include_indexes: bool = True
    include_stats: bool = True
    include_samples: bool = True
    include_query_guidance: bool = True
    include_jsonb_metadata: bool = True
    common_columns_limit: int = 20

    def __post_init__(self):
        """Fail early on values that would produce broken SQL or empty output."""
        if self.sample_rows < 0:
            raise ValueError("sample_rows must be >= 0")
        if self.sample_chars < 1 or self.header_chars < 1:
            raise ValueError("sample_chars and header_chars must be >= 1")
        if any(v < 1 for v in self.column_sample_chars.values()):
            raise ValueError("column_sample_chars values must be >= 1")
        if self.statement_timeout_ms < 0:
            raise ValueError("statement_timeout_ms must be >= 0 (0 = no limit)")
        logger.info("Settings: %s", self)


class PostgresContextGenerator:
    """Generate comprehensive RAG context from a PostgreSQL database for LLM queries."""

    def __init__(
        self,
        connection_string: str,
        tables_filter: Optional[List[str]] = None,
        settings: Optional[ContextSettings] = None,
    ):
        """
        Args:
            connection_string: PostgreSQL URI, e.g. "postgresql://user:pass@localhost:5432/db"
            tables_filter: Optional list of table names to include (None = all tables).
            settings: ContextSettings instance (None = all defaults).
        """
        self.connection_string = connection_string
        self.tables_filter = tables_filter
        self.settings = settings or ContextSettings()
        self.conn = None

    def connect(self):
        """Open a read-only, autocommit connection (a failed query can't poison later ones)."""
        try:
            self.conn = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor,
                options=f"-c statement_timeout={int(self.settings.statement_timeout_ms)}",
            )
            self.conn.set_session(readonly=True, autocommit=True)
            logger.info("Connected to PostgreSQL (read-only, autocommit)")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    @staticmethod
    def _table(schema: str, table: str) -> sql.Composed:
        """Safely quoted schema.table identifier."""
        return sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))

    @staticmethod
    def _quote_name(name: str) -> str:
        """Quote a name only when needed (for readable CREATE TABLE text)."""
        return (
            name
            if re.fullmatch(r"[a-z_][a-z0-9_]*", name)
            else '"' + name.replace('"', '""') + '"'
        )

    @staticmethod
    def _format_cell(value, max_chars: int) -> str:
        """Make any value safe for a one-line markdown table cell."""
        if value is None:
            return "NULL"
        raw = str(value)
        was_cut = len(raw) > max_chars
        text = re.sub(r"\s+", " ", raw[:max_chars]).strip().replace("|", "\\|")
        return text + "…" if was_cut else text

    def _chars_for(self, table: str, column: str) -> int:
        """Cell limit: 'table.column' override > 'column' override > sample_chars."""
        overrides = self.settings.column_sample_chars
        return overrides.get(
            f"{table}.{column}", overrides.get(column, self.settings.sample_chars)
        )

    def _is_hidden(self, table: str, column: str) -> bool:
        hidden = self.settings.exclude_sample_columns
        return f"{table}.{column}" in hidden or column in hidden

    def get_all_tables(self, schema: str = "public") -> List[str]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                (schema,),
            )
            return [row["table_name"] for row in cursor.fetchall()]

    def get_filtered_tables(self, schema: str = "public") -> List[str]:
        all_tables = self.get_all_tables(schema)
        if not self.tables_filter:
            return all_tables
        filtered = [t for t in self.tables_filter if t in all_tables]
        missing = [t for t in self.tables_filter if t not in all_tables]
        if missing:
            logger.warning("Tables not found: %s", ", ".join(missing))
        if filtered:
            logger.info("Using filtered tables: %s", ", ".join(filtered))
            return filtered
        logger.warning("No valid tables in filter, using all tables")
        return all_tables

    def get_all_schemas(self) -> List[str]:
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                  AND schema_name NOT LIKE 'pg_toast%%'
                  AND schema_name NOT LIKE 'pg_temp%%'
                ORDER BY schema_name
                """
            )
            return [row["schema_name"] for row in cursor.fetchall()]

    def get_table_columns(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Columns with exact Postgres type text (format_type), PK flag, and type category."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.attnum AS cid,
                       a.attname AS name,
                       format_type(a.atttypid, a.atttypmod) AS type,
                       t.typname AS base_type,
                       t.typcategory AS category,
                       a.attnotnull AS notnull,
                       pg_get_expr(d.adbin, d.adrelid) AS default_value,
                       COALESCE(a.attnum = ANY(pk.indkey), false) AS pk
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                JOIN pg_type t ON t.oid = a.atttypid
                LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
                LEFT JOIN pg_index pk ON pk.indrelid = a.attrelid AND pk.indisprimary
                WHERE n.nspname = %s AND c.relname = %s
                  AND a.attnum > 0 AND NOT a.attisdropped
                ORDER BY a.attnum
                """,
                (schema, table_name),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_table_schema(
        self,
        table_name: str,
        schema: str = "public",
        columns: Optional[List[Dict]] = None,
    ) -> Optional[str]:
        """Build a CREATE TABLE statement from the column metadata."""
        columns = (
            columns
            if columns is not None
            else self.get_table_columns(table_name, schema)
        )
        if not columns:
            return None
        lines = []
        for col in columns:
            nullable = "NOT NULL" if col["notnull"] else "NULL"
            default = f" DEFAULT {col['default_value']}" if col["default_value"] else ""
            lines.append(
                f"    {self._quote_name(col['name'])} {col['type']} {nullable}{default}"
            )
        pk_cols = [self._quote_name(c["name"]) for c in columns if c["pk"]]
        if pk_cols:
            lines.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")
        return f"CREATE TABLE {schema}.{table_name} (\n" + ",\n".join(lines) + "\n);"

    def get_row_count(self, table_name: str, schema: str = "public") -> Optional[int]:
        """Exact row count, or None if it could not be counted (timeout, permissions...)."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) AS cnt FROM {}").format(
                        self._table(schema, table_name)
                    )
                )
                return cursor.fetchone()["cnt"]
        except Exception as e:
            logger.warning("Row count failed for %s.%s: %s", schema, table_name, e)
            return None

    def get_foreign_keys(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Foreign keys read from pg_constraint (correct for composite keys)."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT a.attname  AS source_column,
                       nf.nspname AS target_schema,
                       cf.relname AS target_table,
                       af.attname AS target_column
                FROM pg_constraint con
                JOIN pg_class c      ON c.oid = con.conrelid
                JOIN pg_namespace n  ON n.oid = c.relnamespace
                JOIN pg_class cf     ON cf.oid = con.confrelid
                JOIN pg_namespace nf ON nf.oid = cf.relnamespace
                JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY AS k(src, tgt, ord) ON true
                JOIN pg_attribute a  ON a.attrelid = con.conrelid  AND a.attnum = k.src
                JOIN pg_attribute af ON af.attrelid = con.confrelid AND af.attnum = k.tgt
                WHERE con.contype = 'f' AND n.nspname = %s AND c.relname = %s
                ORDER BY con.conname, k.ord
                """,
                (schema, table_name),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_indexes(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Indexes with their full definition (handles expression and partial indexes)."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT i.relname AS index_name,
                       ix.indisunique AS is_unique,
                       ix.indisprimary AS is_primary,
                       pg_get_indexdef(ix.indexrelid) AS definition
                FROM pg_index ix
                JOIN pg_class t     ON t.oid = ix.indrelid
                JOIN pg_class i     ON i.oid = ix.indexrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = %s AND t.relname = %s
                ORDER BY i.relname
                """,
                (schema, table_name),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_sample_data(
        self,
        table_name: str,
        schema: str = "public",
        limit: Optional[int] = None,
        columns: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Sample rows. Every value is cut INSIDE PostgreSQL (LEFT(col::text, n+1)),
        so huge text/json/xml/array values never travel to Python.
        bytea is replaced by its size, hidden columns by <hidden>. NULL stays NULL.
        Cell limits come from settings (default + per-column overrides).
        """
        limit = limit if limit is not None else self.settings.sample_rows
        columns = (
            columns
            if columns is not None
            else self.get_table_columns(table_name, schema)
        )
        if not columns:
            return []
        select_parts = []
        for col in columns:
            ident = sql.Identifier(col["name"])
            if self._is_hidden(table_name, col["name"]):
                select_parts.append(sql.SQL("'<hidden>' AS {c}").format(c=ident))
            elif col["base_type"] in BINARY_TYPES:
                select_parts.append(
                    sql.SQL(
                        "'<binary ' || octet_length({c}) || ' bytes>' AS {c}"
                    ).format(c=ident)
                )
            else:
                n = self._chars_for(table_name, col["name"]) + 1
                select_parts.append(
                    sql.SQL("LEFT({c}::text, {n}) AS {c}").format(
                        c=ident, n=sql.Literal(n)
                    )
                )
        query = sql.SQL("SELECT {cols} FROM {tbl} LIMIT {lim}").format(
            cols=sql.SQL(", ").join(select_parts),
            tbl=self._table(schema, table_name),
            lim=sql.Literal(int(limit)),
        )
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                rows = [dict(r) for r in cursor.fetchall()]
        except Exception as e:
            logger.warning("Sampling failed for %s.%s: %s", schema, table_name, e)
            return []
        logger.info("Sampled %d rows from %s.%s", len(rows), schema, table_name)
        return rows

    @staticmethod
    def _distinct_expr(col: Dict) -> sql.Composable:
        """
        Expression used inside COUNT(DISTINCT ...).
        Numbers/dates/booleans: the column itself.
        Everything else (long text, json, xml, arrays...): md5 of its text form, which
        works for types with no equality operator and keeps the sort key small.
        """
        ident = sql.Identifier(col["name"])
        if col["base_type"] in BINARY_TYPES:
            return sql.SQL("md5({})").format(ident)
        if col["category"] in SIMPLE_CATEGORIES:
            return ident
        return sql.SQL("md5({}::text)").format(ident)

    def _stats_query(
        self, table_name: str, schema: str, cols: List[Dict]
    ) -> Dict[str, Dict]:
        """One table scan computing stats for all given columns."""
        parts = [sql.SQL("COUNT(*) AS total")]
        for i, col in enumerate(cols):
            parts.append(
                sql.SQL("COUNT(DISTINCT {e}) AS {d}").format(
                    e=self._distinct_expr(col), d=sql.Identifier(f"d{i}")
                )
            )
            parts.append(
                sql.SQL("COUNT(*) FILTER (WHERE {c} IS NULL) AS {n}").format(
                    c=sql.Identifier(col["name"]), n=sql.Identifier(f"n{i}")
                )
            )
        query = sql.SQL("SELECT {p} FROM {t}").format(
            p=sql.SQL(", ").join(parts), t=self._table(schema, table_name)
        )
        with self.conn.cursor() as cursor:
            cursor.execute(query)
            row = cursor.fetchone()
        total = row["total"]
        return {
            col["name"]: {
                "type": col["type"],
                "total_count": total,
                "distinct_count": row[f"d{i}"],
                "null_count": row[f"n{i}"],
                "null_percentage": round(row[f"n{i}"] / total * 100, 2) if total else 0,
            }
            for i, col in enumerate(cols)
        }

    def get_column_statistics(
        self,
        table_name: str,
        schema: str = "public",
        columns: Optional[List[Dict]] = None,
    ) -> Dict[str, Dict]:
        """
        Distinct/null stats per column. Tries one fast scan for the whole table;
        if that fails, retries column by column so one bad column cannot hide the rest.
        """
        columns = (
            columns
            if columns is not None
            else self.get_table_columns(table_name, schema)
        )
        if not columns:
            return {}
        try:
            return self._stats_query(table_name, schema, columns)
        except Exception as e:
            logger.warning(
                "Table-wide stats failed for %s.%s (%s); retrying per column",
                schema,
                table_name,
                e,
            )
        stats: Dict[str, Dict] = {}
        for col in columns:
            try:
                stats.update(self._stats_query(table_name, schema, [col]))
            except Exception as e:
                logger.warning(
                    "Stats skipped for %s.%s.%s: %s", schema, table_name, col["name"], e
                )
        return stats

    def get_jsonb_sample_structure(
        self, table_name: str, column_name: str, schema: str = "public"
    ) -> Optional[Any]:
        """Extract representative JSON structure from a jsonb column."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT 1"
                    ).format(
                        col=sql.Identifier(column_name),
                        tbl=self._table(schema, table_name),
                    )
                )
                row = cursor.fetchone()
                if row and row[column_name] is not None:
                    return row[column_name]
        except Exception as e:
            logger.warning(
                "Failed to sample JSON structure for %s.%s: %s",
                table_name,
                column_name,
                e,
            )
        return None

    def get_jsonb_key_stats(
        self, table_name: str, column_name: str, schema: str = "public"
    ) -> Dict[str, int]:
        """Count frequency of top-level JSON object keys."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        SELECT key, COUNT(*) AS freq
                        FROM {tbl}, jsonb_object_keys({col}) AS key
                        WHERE {col} IS NOT NULL AND jsonb_typeof({col}) = 'object'
                        GROUP BY key
                        ORDER BY freq DESC
                        LIMIT 20
                        """
                    ).format(
                        tbl=self._table(schema, table_name),
                        col=sql.Identifier(column_name),
                    )
                )
                return {row["key"]: row["freq"] for row in cursor.fetchall()}
        except Exception as e:
            logger.warning(
                "Failed to analyze JSON keys for %s.%s: %s",
                table_name,
                column_name,
                e,
            )
            return {}

    def _infer_json_type(self, value: Any) -> str:
        """Infer the PostgreSQL/JSON type of a value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "numeric"
        if isinstance(value, str):
            return "text"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"

    def get_jsonb_key_types(
        self, table_name: str, column_name: str, schema: str = "public"
    ) -> Dict[str, str]:
        """
        Infer the type of each top-level key in a jsonb object column
        using a single aggregated query. Ignores null values to find the
        actual data type used when the field is populated.
        """
        try:
            with self.conn.cursor() as cursor:
                # Extract all keys and their jsonb_typeof from non-null jsonb objects
                # We scan up to 5000 rows to ensure we catch sparse fields
                cursor.execute(
                    sql.SQL("""
                        SELECT key, jsonb_typeof(value) as type
                        FROM {tbl}, jsonb_each({col})
                        WHERE {col} IS NOT NULL AND jsonb_typeof({col}) = 'object'
                        LIMIT 5000
                    """).format(
                        tbl=self._table(schema, table_name),
                        col=sql.Identifier(column_name),
                    )
                )

                rows = cursor.fetchall()
                key_types: Dict[str, Set[str]] = {}

                for row in rows:
                    key = row["key"]
                    pg_type = row["type"]

                    if key not in key_types:
                        key_types[key] = set()
                    key_types[key].add(pg_type)

                # Resolve final types
                resolved_types: Dict[str, str] = {}
                type_map = {
                    "string": "text",
                    "number": "numeric",
                    "boolean": "boolean",
                    "array": "array",
                    "object": "object",
                }

                for key, types in key_types.items():
                    # Remove 'null' from consideration to find the actual data type
                    clean_types = types - {"null"}

                    if not clean_types:
                        # If only nulls were found, mark as null
                        resolved_types[key] = "null"
                    elif len(clean_types) == 1:
                        pg_t = clean_types.pop()
                        resolved_types[key] = type_map.get(pg_t, "unknown")
                    else:
                        # Mixed types: prefer text if string is involved, else numeric
                        if "string" in clean_types:
                            resolved_types[key] = "text"
                        elif "number" in clean_types:
                            resolved_types[key] = "numeric"
                        else:
                            resolved_types[key] = "mixed"

                return resolved_types

        except Exception as e:
            logger.warning(
                "Failed to analyze JSON key types for %s.%s: %s",
                table_name,
                column_name,
                e,
            )
            return {}

    def _format_json_value(self, value: Any, max_chars: int = 50) -> Any:
        """
        Recursively format JSON values for display.
        - Keeps native types (bool, int, float, null) so json.dumps renders them correctly.
        - Truncates strings.
        - Truncates lists to first 3 items.
        """
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, str):
            if len(value) > max_chars:
                return value[:max_chars] + "..."
            return value
        if isinstance(value, list):
            if not value:
                return []
            items = [self._format_json_value(v, max_chars) for v in value[:3]]
            if len(value) > 3:
                items.append("...")
            return items
        if isinstance(value, dict):
            if not value:
                return {}
            result = {}
            keys = list(value.keys())[:5]
            for k in keys:
                result[k] = self._format_json_value(value[k], max_chars)
            if len(value) > 5:
                result["..."] = "..."
            return result
        return str(value)

    def get_jsonb_metadata(self, table_name: str, schema: str = "public") -> List[str]:
        """Generate metadata lines for all jsonb columns in a table."""
        columns = self.get_table_columns(table_name, schema)
        jsonb_cols = [c for c in columns if c["base_type"] == "jsonb"]

        if not jsonb_cols:
            return []

        lines: List[str] = [f"\n### JSONB Columns in {schema}.{table_name}"]
        for col in jsonb_cols:
            col_name = col["name"]
            struct = self.get_jsonb_sample_structure(table_name, col_name, schema)
            key_types = self.get_jsonb_key_types(table_name, col_name, schema)

            lines.append(f"\n**`{col_name}`**:")
            if struct is not None:
                if isinstance(struct, dict):
                    lines.append("- Type: Object")

                    if key_types:
                        sorted_keys = sorted(key_types.items())
                        type_strings = [f"{k} ({v})" for k, v in sorted_keys]
                        lines.append(f"- Keys & Types: {', '.join(type_strings)}")

                    formatted_sample = {}
                    for k, v in struct.items():
                        formatted_sample[k] = self._format_json_value(v, max_chars=80)

                    try:
                        sample_str = json.dumps(
                            formatted_sample, indent=2, ensure_ascii=False
                        )
                        lines.append(f"- Sample Structure:\n```json\n{sample_str}\n```")
                    except (TypeError, ValueError):
                        lines.append(f"- Sample: {str(struct)[:200]}")

                elif isinstance(struct, list):
                    lines.append("- Type: Array")
                    if struct:
                        elem_type = self._infer_json_type(struct[0])
                        lines.append(f"- Element type: {elem_type}")
                        preview = [self._format_json_value(e, 60) for e in struct[:3]]
                        lines.append(f"- Sample elements: {preview}")
                    else:
                        lines.append("- Empty array in sample")
                else:
                    lines.append(f"- Type: {self._infer_json_type(struct)}")
                    lines.append(f"- Sample: {str(struct)[:200]}")
            else:
                lines.append("- No non-null samples found")

        return lines

    def generate_rag_context(self, schema: str = "public") -> str:
        """Generate comprehensive RAG context string for the LLM."""
        logger.info("Generating context for schema '%s'", schema)
        self.connect()
        try:
            return self._build_context(schema)
        finally:
            self.disconnect()

    def _build_context(self, schema: str) -> str:
        cfg = self.settings
        out: List[str] = [
            "=" * 80,
            "POSTGRESQL DATABASE SCHEMA & METADATA CONTEXT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 80,
            "",
        ]
        schemas = self.get_all_schemas()
        tables = self.get_filtered_tables(schema)
        cols_by_table = {t: self.get_table_columns(t, schema) for t in tables}
        out += [
            "## DATABASE OVERVIEW",
            f"Available Schemas: {', '.join(schemas)}",
            f"Target Schema: {schema}",
            f"Total Tables: {len(tables)}",
            f"Tables: {', '.join(tables)}",
        ]
        if self.tables_filter:
            out.append(f"Filter Applied: {', '.join(self.tables_filter)}")
        out += ["", "## TABLE SCHEMAS", "-" * 80]
        for table in tables:
            columns = cols_by_table[table]
            count = self.get_row_count(table, schema)
            out.append(
                f"\n### Table: {schema}.{table} ({'?' if count is None else count} rows)"
            )
            schema_sql = self.get_table_schema(table, schema, columns)
            if schema_sql:
                out.append(f"```sql\n{schema_sql}\n```")
            out.append("\n**Columns:**")
            for col in columns:
                pk = " [PK]" if col["pk"] else ""
                nullable = "" if col["notnull"] else " (nullable)"
                out.append(f"- `{col['name']}`: {col['type']}{pk}{nullable}")
            fks = (
                self.get_foreign_keys(table, schema) if cfg.include_foreign_keys else []
            )
            if fks:
                out.append("\n**Foreign Keys:**")
                for fk in fks:
                    out.append(
                        f"- `{fk['source_column']}` → "
                        f"`{fk['target_schema']}.{fk['target_table']}.{fk['target_column']}`"
                    )
            idxs = self.get_indexes(table, schema) if cfg.include_indexes else []
            if idxs:
                out.append("\n**Indexes:**")
                for idx in idxs:
                    flags = (" [UNIQUE]" if idx["is_unique"] else "") + (
                        " [PRIMARY]" if idx["is_primary"] else ""
                    )
                    out.append(f"- `{idx['index_name']}`{flags}: {idx['definition']}")
        stats_tables: List[str] = []
        if cfg.include_stats:
            out += ["", "## COLUMN STATISTICS SUMMARY", "-" * 80]
            stats_tables = (
                tables
                if cfg.max_stats_tables is None
                else tables[: cfg.max_stats_tables]
            )
        for table in stats_tables:
            col_stats = self.get_column_statistics(table, schema, cols_by_table[table])
            if not col_stats:
                continue
            out += [
                f"\n### Table: {table}",
                "| Column | Type | Distinct Values | Null % |",
                "|--------|------|-----------------|--------|",
            ]
            for name, s in col_stats.items():
                out.append(
                    f"| `{self._format_cell(name, 200)}` | {s['type']} | {s['distinct_count']} | {s['null_percentage']}% |"
                )

        # JSONB metadata section
        if cfg.include_jsonb_metadata:
            has_jsonb = any(
                any(c["base_type"] == "jsonb" for c in cols_by_table[t]) for t in tables
            )
            if has_jsonb:
                out += ["", "## JSONB COLUMN METADATA", "-" * 80]
                for table in tables:
                    jsonb_meta = self.get_jsonb_metadata(table, schema)
                    out += jsonb_meta

        sample_tables = tables if cfg.include_samples and cfg.sample_rows > 0 else []
        if sample_tables:
            out += ["", "## SAMPLE DATA", "-" * 80]
        for table in sample_tables:
            samples = self.get_sample_data(table, schema, columns=cols_by_table[table])
            if not samples:
                continue
            names = list(samples[0].keys())
            out += [
                f"\n### Table: {table} (first {len(samples)} rows)",
                "| "
                + " | ".join(self._format_cell(n, cfg.header_chars) for n in names)
                + " |",
                "| " + " | ".join(["---"] * len(names)) + " |",
            ]
            for row in samples:
                cells = (
                    self._format_cell(row[n], self._chars_for(table, n)) for n in names
                )
                out.append("| " + " | ".join(cells) + " |")
        common = {}
        for table in tables:
            for col in cols_by_table[table]:
                common.setdefault(col["name"], []).append(f"{table}.{col['type']}")
        limit = cfg.common_columns_limit
        common_cols = ", ".join(list(common)[:limit]) + (
            "..." if len(common) > limit else ""
        )
        guidance = [
            "",
            "## QUERY GUIDANCE FOR DYNAMIC POSTGRESQL QUERIES",
            "-" * 80,
            "",
            "When constructing PostgreSQL queries based on user questions:",
            "",
            "1. **Identify Tables**: Use table names above to determine which tables to query",
            "2. **Available Columns**: See column lists for each table above",
            f"3. **Common Column Names Found**: {common_cols}",
            "4. **Relationships**: Check foreign keys section for table relationships",
            "5. **Data Types**: Note column types for proper filtering and casting",
            "6. **Truncated values**: Sample values ending with … are cut for display only",
            "",
            "Example Query Pattern:",
            "```python",
            "import psycopg2",
            "",
            'conn = psycopg2.connect("<CONNECTION_STRING>")  # credentials intentionally omitted',
            "cursor = conn.cursor()",
            "",
            "# Simple SELECT with filtering",
            'cursor.execute("""',
            f"    SELECT * FROM {schema}.{{table_name}}",
            "    WHERE {column} = %s",
            "    LIMIT 10",
            '""", (value,))',
            "",
            "results = cursor.fetchall()",
            "```",
            "",
            "**Best Practices:**",
            "- Always use parameterized queries (%s) to prevent SQL injection",
            f"- Use explicit schema qualification ({schema}.table_name)",
            "- Leverage indexes shown above for better performance",
            "- Consider NULL handling in filters (IS NULL / IS NOT NULL)",
            "- Use LIMIT for large tables to avoid memory issues",
        ]

        # Add JSONB-specific query patterns if jsonb columns exist
        if cfg.include_jsonb_metadata and has_jsonb:
            guidance += [
                "",
                "**JSONB Query Patterns (GIN index supported):**",
                "```sql",
                "-- Check if jsonb array contains a value",
                f"SELECT * FROM {schema}.jobs WHERE tags @> '[\"remote\"]'::jsonb;",
                "",
                "-- Check if jsonb object has a key",
                f"SELECT * FROM {schema}.job_entities WHERE entities ? 'work_mode';",
                "",
                "-- Check if jsonb object has key-value pair",
                f'SELECT * FROM {schema}.job_entities WHERE entities @> \'{{"work_mode": "remote"}}\'::jsonb;',
                "",
                "-- Extract specific value from jsonb",
                f"SELECT entities->>'work_mode' AS work_mode FROM {schema}.job_entities;",
                "```",
            ]

        if cfg.include_query_guidance:
            out += guidance
        out.append("=" * 80)
        logger.info("Context built: %d tables, %d lines", len(tables), len(out))
        return "\n".join(out)


def generate_rag_context(
    connection_string: str,
    schema: str = "public",
    tables_filter: Optional[List[str]] = None,
    settings: Optional[ContextSettings] = None,
) -> str:
    """Standalone helper. Pass a ContextSettings to change defaults."""
    generator = PostgresContextGenerator(
        connection_string, tables_filter=tables_filter, settings=settings
    )
    return generator.generate_rag_context(schema=schema)


DEFAULT_DB_URL = "postgresql://jethroestrada:@localhost:5432/jobs_db3"
DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "generated"
    / Path(__file__).stem
    / "postgres_db_context.txt"
)


def _parse_limit(value: str) -> Optional[int]:
    """'all' (or 'none') -> None (no limit), otherwise a positive integer."""
    if value.lower() in {"all", "none"}:
        return None
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number or 'all', got '{value}'")
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1 (or 'all')")
    return number


def _parse_column_limit(value: str):
    """'description=300' or 'jobs.title=100' -> ('description', 300)."""
    name, sep, number = value.rpartition("=")
    if not sep or not name or not number.isdigit() or int(number) < 1:
        raise argparse.ArgumentTypeError(
            f"expected COLUMN=CHARS (e.g. description=300), got '{value}'"
        )
    return name, int(number)


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments. Defaults come from ContextSettings, so the CLI and
    the Python API always agree. Returns a Namespace with a ready-to-use `args.settings`.
    """
    d = ContextSettings()
    parser = argparse.ArgumentParser(
        prog="postgres_context_generator",
        description="Generate PostgreSQL schema/sample context text for LLM (RAG) prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                        # all defaults\n"
            "  %(prog)s -t jobs,entities -r 5 -c 80            # two tables, 5 rows, 80 chars\n"
            "  %(prog)s -C description=300 -C jobs.title=100   # per-column cell limits\n"
            "  %(prog)s -x email,users.phone -m all            # hide columns, stats for all tables\n"
            "  %(prog)s --no-indexes --no-guidance -p 0        # smaller output, no preview\n"
        ),
    )
    conn = parser.add_argument_group("connection and output")
    conn.add_argument(
        "-u",
        "--db-url",
        default=None,
        help="PostgreSQL URI (default: $DATABASE_URL, else the local jobs_db3 URI)",
    )
    conn.add_argument(
        "-s",
        "--schema",
        default="public",
        help="Schema to analyze (default: %(default)s)",
    )
    conn.add_argument(
        "-t",
        "--tables",
        type=str,
        metavar="TABLES",
        default=None,
        help="Comma-separated list of tables to include (default: all tables)",
    )
    conn.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path (default: %(default)s)",
    )
    conn.add_argument(
        "-p",
        "--preview-chars",
        type=int,
        default=2000,
        help="Characters of the result printed to the terminal, 0 = none (default: %(default)s)",
    )
    conn.add_argument(
        "-l",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: %(default)s)",
    )
    samples = parser.add_argument_group("sample data")
    samples.add_argument(
        "-r",
        "--sample-rows",
        type=int,
        default=d.sample_rows,
        help="Sample rows per table, 0 = none (default: %(default)s)",
    )
    samples.add_argument(
        "-c",
        "--sample-chars",
        type=int,
        default=d.sample_chars,
        help="Max characters per sample cell (default: %(default)s)",
    )
    samples.add_argument(
        "-H",
        "--header-chars",
        type=int,
        default=d.header_chars,
        help="Max characters per sample column header (default: %(default)s)",
    )
    samples.add_argument(
        "-C",
        "--column-chars",
        dest="column_sample_chars",
        action="append",
        type=_parse_column_limit,
        metavar="COLUMN=CHARS",
        default=None,
        help="Cell limit for one column; repeatable. Use 'table.column' to target one table",
    )
    samples.add_argument(
        "-x",
        "--hide-columns",
        dest="exclude_sample_columns",
        nargs="+",
        metavar="COLUMN",
        default=None,
        help="Show <hidden> instead of values for these columns ('column' or 'table.column')",
    )
    stats = parser.add_argument_group("statistics and safety")
    stats.add_argument(
        "-m",
        "--max-stats-tables",
        type=_parse_limit,
        default=d.max_stats_tables,
        metavar="N|all",
        help="Tables that get column statistics (default: %(default)s)",
    )
    stats.add_argument(
        "-T",
        "--timeout-ms",
        dest="statement_timeout_ms",
        type=int,
        default=d.statement_timeout_ms,
        help="Per-query time limit in milliseconds, 0 = no limit (default: %(default)s)",
    )
    stats.add_argument(
        "-n",
        "--common-columns",
        dest="common_columns_limit",
        type=int,
        default=d.common_columns_limit,
        help="Column names listed in the guidance section (default: %(default)s)",
    )
    sections = parser.add_argument_group("sections (turn off with --no-<name>)")
    for flag, name, label in [
        ("foreign-keys", "include_foreign_keys", "foreign keys"),
        ("indexes", "include_indexes", "indexes"),
        ("stats", "include_stats", "column statistics"),
        ("samples", "include_samples", "sample data"),
        ("guidance", "include_query_guidance", "query guidance"),
        ("jsonb-metadata", "include_jsonb_metadata", "JSONB metadata"),
    ]:
        sections.add_argument(
            f"--{flag}",
            dest=name,
            action=argparse.BooleanOptionalAction,
            default=getattr(d, name),
            help=f"Include {label} (default: %(default)s)",
        )
    args = parser.parse_args(argv)

    # Parse comma-separated tables if provided
    if args.tables:
        args.tables = [tbl.strip() for tbl in args.tables.split(",") if tbl.strip()]

    try:
        args.settings = build_settings(args)
    except ValueError as e:
        parser.error(str(e))
    return args


def build_settings(args: argparse.Namespace) -> ContextSettings:
    """Turn parsed arguments into ContextSettings (argument names match the field names)."""
    values = {f.name: getattr(args, f.name) for f in fields(ContextSettings)}
    values["column_sample_chars"] = dict(values["column_sample_chars"] or [])
    values["exclude_sample_columns"] = set(values["exclude_sample_columns"] or [])
    return ContextSettings(**values)


def main(argv: Optional[List[str]] = None) -> None:
    args = get_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("Settings: %s", args.settings)
    db_url = args.db_url or os.getenv("DATABASE_URL") or DEFAULT_DB_URL
    context = generate_rag_context(
        db_url, schema=args.schema, tables_filter=args.tables, settings=args.settings
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(context)
    if args.preview_chars > 0:
        print(
            f"\nPreview (first {args.preview_chars} chars):\n{context[: args.preview_chars]}..."
        )
    logger.info("Context saved to %s", args.output)


if __name__ == "__main__":
    main()
