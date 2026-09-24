from datetime import datetime
from typing import Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresContextGenerator:
    """Generate comprehensive RAG context from PostgreSQL database for LLM queries."""

    def __init__(self, connection_string: str):
        """
        Initialize with PostgreSQL connection string.

        Args:
            connection_string: PostgreSQL connection URI
                              (e.g., "postgresql://user:pass@localhost:5432/dbname")
        """
        self.connection_string = connection_string
        self.conn = None

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                self.connection_string, cursor_factory=RealDictCursor
            )
            print(f"✓ Connected to PostgreSQL database")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")

    def get_all_tables(self, schema: str = "public") -> List[str]:
        """Get list of all tables in specified schema."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """,
                (schema,),
            )
            return [row["table_name"] for row in cursor.fetchall()]

    def get_all_schemas(self) -> List[str]:
        """Get list of all schemas in the database."""
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schema_name
            """)
            return [row["schema_name"] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str, schema: str = "public") -> str:
        """Get CREATE TABLE statement for a specific table."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    is_nullable,
                    column_default,
                    ordinal_position
                FROM information_schema.columns
                WHERE table_schema = %s 
                AND table_name = %s
                ORDER BY ordinal_position
            """,
                (schema, table_name),
            )

            columns = cursor.fetchall()

            if not columns:
                return None

            lines = [f"CREATE TABLE {schema}.{table_name} ("]

            for i, col in enumerate(columns):
                type_str = col["data_type"]

                if col["character_maximum_length"]:
                    type_str += f"({col['character_maximum_length']})"
                elif (
                    col["numeric_precision"] and col["data_type"] != "double precision"
                ):
                    if col["numeric_scale"]:
                        type_str += (
                            f"({col['numeric_precision']},{col['numeric_scale']})"
                        )
                    else:
                        type_str += f"({col['numeric_precision']})"

                nullable = "NOT NULL" if col["is_nullable"] == "NO" else "NULL"
                default = (
                    f" DEFAULT {col['column_default']}" if col["column_default"] else ""
                )

                comma = "," if i < len(columns) - 1 else ""
                lines.append(
                    f"    {col['column_name']} {type_str} {nullable}{default}{comma}"
                )

            # Get primary keys
            cursor.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = %s 
                AND tc.table_name = %s 
                AND tc.constraint_type = 'PRIMARY KEY'
            """,
                (schema, table_name),
            )

            pk_columns = [row["column_name"] for row in cursor.fetchall()]

            if pk_columns:
                lines[-1] = lines[-1].rstrip(",") + ","
                pk_str = ", ".join(pk_columns)
                lines.append(f"    PRIMARY KEY ({pk_str})")

            lines.append(");")

            return "\n".join(lines)

    def get_table_columns(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Get detailed column information for a table."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT 
                    column_name as name,
                    data_type as type,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    CASE WHEN is_nullable = 'NO' THEN 1 ELSE 0 END as notnull,
                    column_default as default_value,
                    ordinal_position as cid,
                    CASE 
                        WHEN EXISTS (
                            SELECT 1 FROM information_schema.key_column_usage kcu
                            JOIN information_schema.table_constraints tc 
                                ON kcu.constraint_name = tc.constraint_name
                            WHERE kcu.column_name = c.column_name
                            AND kcu.table_schema = c.table_schema
                            AND kcu.table_name = c.table_name
                            AND tc.constraint_type = 'PRIMARY KEY'
                        ) THEN 1 
                        ELSE 0 
                    END as pk
                FROM information_schema.columns c
                WHERE c.table_schema = %s 
                AND c.table_name = %s
                ORDER BY c.ordinal_position
            """,
                (schema, table_name),
            )

            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "cid": row["cid"],
                        "name": row["name"],
                        "type": row["type"],
                        "notnull": row["notnull"],
                        "default_value": row["default_value"],
                        "pk": row["pk"],
                        "max_length": row["character_maximum_length"],
                        "precision": row["numeric_precision"],
                        "scale": row["numeric_scale"],
                    }
                )
            return columns

    def get_row_count(self, table_name: str, schema: str = "public") -> int:
        """Get row count for a specific table."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f'SELECT COUNT(*) as cnt FROM "{schema}"."{table_name}"')
                return cursor.fetchone()["cnt"]
        except Exception:
            return 0

    def get_foreign_keys(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Get foreign key relationships for a table."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    kcu.column_name as source_column,
                    ccu.table_schema as target_schema,
                    ccu.table_name as target_table,
                    ccu.column_name as target_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema = %s
                AND tc.table_name = %s
                AND tc.constraint_type = 'FOREIGN KEY'
            """,
                (schema, table_name),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_indexes(self, table_name: str, schema: str = "public") -> List[Dict]:
        """Get index information for a table."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    i.relname as index_name,
                    ix.indisunique as is_unique,
                    ix.indisprimary as is_primary,
                    array_agg(a.attname ORDER by k.n) as columns
                FROM pg_class t
                JOIN pg_namespace n ON t.relnamespace = n.oid
                JOIN pg_index ix ON t.oid = ix.indrelid
                JOIN pg_class i ON ix.indexrelid = i.oid
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                JOIN generate_subscripts(ix.indkey, 1) k(n) ON true
                WHERE n.nspname = %s
                AND t.relname = %s
                GROUP BY i.relname, ix.indisunique, ix.indisprimary
                ORDER BY i.relname
            """,
                (schema, table_name),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_sample_data(
        self, table_name: str, schema: str = "public", limit: int = 5
    ) -> List[Dict]:
        """Get sample rows from a table."""
        with self.conn.cursor() as cursor:
            cursor.execute(
                f'SELECT * FROM "{schema}"."{table_name}" LIMIT %s', (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_column_statistics(
        self, table_name: str, schema: str = "public"
    ) -> Dict[str, Dict]:
        """Get basic statistics for each column (distinct values, nulls, etc.)."""
        stats = {}
        columns = self.get_table_columns(table_name, schema)

        for col in columns:
            col_name = col["name"]
            col_type = col["type"]

            with self.conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT 
                        COUNT(DISTINCT "{col_name}") as distinct_count,
                        COUNT(*) FILTER (WHERE "{col_name}" IS NULL) as null_count,
                        COUNT(*) as total_count
                    FROM "{schema}"."{table_name}"
                """)

                result = cursor.fetchone()
                if result:
                    stats[col_name] = {
                        "type": col_type,
                        "total_count": result["total_count"],
                        "distinct_count": result["distinct_count"],
                        "null_count": result["null_count"],
                        "null_percentage": round(
                            (result["null_count"] / result["total_count"] * 100)
                            if result["total_count"] > 0
                            else 0,
                            2,
                        ),
                    }

        return stats

    def generate_full_context(self, schema: str = "public") -> str:
        """Generate comprehensive RAG context string for LLM."""
        self.connect()

        context_parts = []

        context_parts.append("=" * 80)
        context_parts.append("POSTGRESQL DATABASE SCHEMA & METADATA CONTEXT")
        context_parts.append(f"Generated: {datetime.now().isoformat()}")
        context_parts.append("=" * 80)
        context_parts.append("")

        schemas = self.get_all_schemas()
        tables = self.get_all_tables(schema)

        context_parts.append("## DATABASE OVERVIEW")
        context_parts.append(f"Available Schemas: {', '.join(schemas)}")
        context_parts.append(f"Target Schema: {schema}")
        context_parts.append(f"Total Tables: {len(tables)}")
        context_parts.append(f"Tables: {', '.join(tables)}")
        context_parts.append("")

        context_parts.append("## TABLE SCHEMAS")
        context_parts.append("-" * 80)

        for table in tables:
            schema_sql = self.get_table_schema(table, schema)
            count = self.get_row_count(table, schema)
            columns = self.get_table_columns(table, schema)
            foreign_keys = self.get_foreign_keys(table, schema)
            indexes = self.get_indexes(table, schema)

            context_parts.append(f"\n### Table: {schema}.{table} ({count} rows)")

            if schema_sql:
                context_parts.append(f"```sql\n{schema_sql}\n```")

            context_parts.append("\n**Columns:**")
            for col in columns:
                pk_marker = " [PK]" if col["pk"] else ""
                nullable = "" if col["notnull"] else " (nullable)"
                type_detail = col["type"]

                if col["max_length"]:
                    type_detail += f"({col['max_length']})"
                elif col["precision"]:
                    if col["scale"]:
                        type_detail += f"({col['precision']},{col['scale']})"
                    else:
                        type_detail += f"({col['precision']})"

                context_parts.append(
                    f"- `{col['name']}`: {type_detail}{pk_marker}{nullable}"
                )

            if foreign_keys:
                context_parts.append("\n**Foreign Keys:**")
                for fk in foreign_keys:
                    context_parts.append(
                        f"- `{fk['source_column']}` → "
                        f"`{fk['target_schema']}.{fk['target_table']}.{fk['target_column']}`"
                    )

            if indexes:
                context_parts.append("\n**Indexes:**")
                for idx in indexes:
                    unique_marker = " [UNIQUE]" if idx["is_unique"] else ""
                    primary_marker = " [PRIMARY]" if idx["is_primary"] else ""
                    cols = ", ".join(idx["columns"])
                    context_parts.append(
                        f"- `{idx['index_name']}`: ({cols}){unique_marker}{primary_marker}"
                    )

        context_parts.append("")

        context_parts.append("## COLUMN STATISTICS SUMMARY")
        context_parts.append("-" * 80)

        for table in tables[:5]:
            col_stats = self.get_column_statistics(table, schema)

            if col_stats:
                context_parts.append(f"\n### Table: {table}")
                context_parts.append("| Column | Type | Distinct Values | Null % |")
                context_parts.append("|--------|------|-----------------|--------|")

                for col_name, stats in col_stats.items():
                    context_parts.append(
                        f"| `{col_name}` | {stats['type']} | "
                        f"{stats['distinct_count']} | {stats['null_percentage']}% |"
                    )

        context_parts.append("")

        context_parts.append("## SAMPLE DATA")
        context_parts.append("-" * 80)

        for table in tables[:3]:
            samples = self.get_sample_data(table, schema, limit=3)

            if samples:
                context_parts.append(f"\n### Table: {table} (first 3 rows)")

                columns = list(samples[0].keys())

                header = "| " + " | ".join(columns) + " |"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"

                context_parts.append(header)
                context_parts.append(separator)

                for row in samples:
                    values = [str(row[col])[:50] for col in columns]
                    context_parts.append("| " + " | ".join(values) + " |")

        context_parts.append("")

        # Query guidance section
        context_parts.append("## QUERY GUIDANCE FOR DYNAMIC POSTGRESQL QUERIES")
        context_parts.append("-" * 80)

        all_columns = {}
        for table in tables:
            columns = self.get_table_columns(table, schema)
            for col in columns:
                if col["name"] not in all_columns:
                    all_columns[col["name"]] = []
                all_columns[col["name"]].append(f"{table}.{col['type']}")

        common_cols = ", ".join(list(all_columns.keys())[:20])
        if len(all_columns) > 20:
            common_cols += "..."

        guidance_lines = [
            "",
            "When constructing PostgreSQL queries based on user questions:",
            "",
            "1. **Identify Tables**: Use table names above to determine which tables to query",
            "2. **Available Columns**: See column lists for each table above",
            f"3. **Common Column Names Found**: {common_cols}",
            "4. **Relationships**: Check foreign keys section for table relationships",
            "5. **Data Types**: Note column types for proper filtering and casting",
            "",
            "Example Query Pattern:",
            "```python",
            "import psycopg2",
            "",
            f'conn = psycopg2.connect("{self.connection_string}")',
            "cursor = conn.cursor()",
            "",
            "# Simple SELECT with filtering",
            'cursor.execute("""',
            f"    SELECT * FROM {schema}.{{table_name}}",
            "    WHERE {{column}} = %s",
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

        context_parts.extend(guidance_lines)

        context_parts.append("=" * 80)

        self.disconnect()

        return "\n".join(context_parts)


if __name__ == "__main__":
    import shutil
    from pathlib import Path

    OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CONNECTION_STRING = "postgresql://jethroestrada:@localhost:5432/jobs_db3"

    generator = PostgresContextGenerator(CONNECTION_STRING)
    context = generator.generate_full_context(schema="public")

    output_file = OUTPUT_DIR / "postgres_db_context.txt"
    with open(output_file, "w") as f:
        f.write(context)

    print(f"\nPreview (first 2000 chars):\n{context[:2000]}...")
    print(f"\n✓ Context saved to {output_file}")
