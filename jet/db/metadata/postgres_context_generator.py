import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Postgres type categories (pg_type.typcategory) that are cheap and safe to COUNT(DISTINCT).
# N = numeric, D = date/time, B = boolean. Everything else is hashed via md5(col::text).
SIMPLE_CATEGORIES = {"N", "D", "B"}
BINARY_TYPES = {"bytea"}


@dataclass
class ContextSettings:
    """All tunable options in one place. Defaults match the original behavior."""

    # --- Sample data
    sample_rows: int = 3  # rows shown per table
    sample_chars: int = 50  # max characters per cell (longer values end with "…")
    header_chars: int = 60  # max characters per column header in sample tables
    # Per-column cell limits. Keys: "column" (any table) or "table.column" (most specific wins)
    column_sample_chars: Dict[str, int] = field(default_factory=dict)
    # Columns whose values are replaced by <hidden>. Keys: "column" or "table.column"
    exclude_sample_columns: Set[str] = field(default_factory=set)

    # --- Statistics
    max_stats_tables: Optional[int] = 5  # None = all tables

    # --- Safety
    statement_timeout_ms: int = 30000  # per-query time limit

    # --- Sections on/off
    include_foreign_keys: bool = True
    include_indexes: bool = True
    include_stats: bool = True
    include_samples: bool = True
    include_query_guidance: bool = True
    common_columns_limit: int = 20  # names listed in the guidance section

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

    # ------------------------------------------------------------------ connection
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

    # ------------------------------------------------------------------ helpers
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

    # ------------------------------------------------------------------ metadata
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

    # ------------------------------------------------------------------ sample data
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
                # n+1 characters lets _format_cell know the value was cut
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

    # ------------------------------------------------------------------ statistics
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

    # ------------------------------------------------------------------ context builder
    def generate_rag_context(self, schema: str = "public") -> str:
        """Generate comprehensive RAG context string for the LLM."""
        logger.info("Generating context for schema '%s'", schema)
        self.connect()
        try:
            return self._build_context(schema)
        finally:
            self.disconnect()

    def _build_context(self, schema: str) -> str:
        s = self.settings
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

            fks = self.get_foreign_keys(table, schema) if s.include_foreign_keys else []
            if fks:
                out.append("\n**Foreign Keys:**")
                for fk in fks:
                    out.append(
                        f"- `{fk['source_column']}` → "
                        f"`{fk['target_schema']}.{fk['target_table']}.{fk['target_column']}`"
                    )

            idxs = self.get_indexes(table, schema) if s.include_indexes else []
            if idxs:
                out.append("\n**Indexes:**")
                for idx in idxs:
                    flags = (" [UNIQUE]" if idx["is_unique"] else "") + (
                        " [PRIMARY]" if idx["is_primary"] else ""
                    )
                    out.append(f"- `{idx['index_name']}`{flags}: {idx['definition']}")

        stats_tables: List[str] = []
        if s.include_stats:
            out += ["", "## COLUMN STATISTICS SUMMARY", "-" * 80]
            stats_tables = (
                tables if s.max_stats_tables is None else tables[: s.max_stats_tables]
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
                    f"| `{name}` | {s['type']} | {s['distinct_count']} | {s['null_percentage']}% |"
                )

        sample_tables = tables if s.include_samples and s.sample_rows > 0 else []
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
                + " | ".join(self._format_cell(n, s.header_chars) for n in names)
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
        limit = s.common_columns_limit
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
        if s.include_query_guidance:
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


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer an environment variable so the password is not stored in code
    CONNECTION_STRING = os.getenv(
        "DATABASE_URL", "postgresql://jethroestrada:@localhost:5432/jobs_db3"
    )
    TABLES_FILTER = None  # e.g. ["jobs", "entities"]; None = all tables

    # Defaults = 3 sample rows, 50 chars per cell, stats for first 5 tables. Override as needed:
    SETTINGS = ContextSettings(
        # sample_rows=5,
        # sample_chars=80,
        # column_sample_chars={"description": 300, "jobs.title": 100},
        # exclude_sample_columns={"email", "users.phone"},
        # max_stats_tables=None,
        # include_indexes=False,
    )

    context = generate_rag_context(
        CONNECTION_STRING,
        schema="public",
        tables_filter=TABLES_FILTER,
        settings=SETTINGS,
    )
    output_file = OUTPUT_DIR / "postgres_db_context.txt"
    output_file.write_text(context)

    print(f"\nPreview (first 2000 chars):\n{context[:2000]}...")
    logger.info("Context saved to %s", output_file)
