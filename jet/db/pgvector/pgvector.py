import psycopg
from pgvector.psycopg import register_vector
import numpy as np


class PgVectorClient:
    def __init__(self, dbname, user, password, host='localhost', port=5432):
        """Initialize the database connection."""
        self.conn = psycopg.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=True
        )
        register_vector(self.conn)
        self._initialize_extension()

    def _initialize_extension(self):
        """Ensure the pgvector extension is enabled."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def create_table(self, table_name, dimension):
        """Create a table with a vector column."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            embedding VECTOR({dimension})
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(query)

    def insert_vector(self, table_name, vector):
        """Insert a vector into the table."""
        query = f"INSERT INTO {table_name} (embedding) VALUES (%s) RETURNING id;"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector,))
            return cur.fetchone()[0]

    def get_vector_by_id(self, table_name, vector_id):
        """Retrieve a vector by its ID."""
        query = f"SELECT embedding FROM {table_name} WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id,))
            result = cur.fetchone()
            return np.array(result[0]) if result else None

    def get_vectors_by_ids(self, table_name, vector_ids):
        """Retrieve multiple vectors by their IDs."""
        query = f"SELECT id, embedding FROM {table_name} WHERE id = ANY(%s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids,))
            results = cur.fetchall()
            return {row[0]: np.array(row[1]) for row in results}

    def update_vector_by_id(self, table_name, vector_id, new_vector):
        """Update a vector by its ID."""
        query = f"UPDATE {table_name} SET embedding = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (new_vector, vector_id))

    def search_similar(self, table_name, query_vector, top_k=5):
        """Find the top-K most similar vectors using L2 distance."""
        query = f"""
        SELECT id, embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {top_k};
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (query_vector,))
            return cur.fetchall()

    def drop_all_rows(self, table_name):
        """Delete all rows from a table."""
        query = f"DELETE FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)

    def delete_all_tables(self):
        """Drop all tables in the database."""
        query = """
        DO $$ DECLARE 
            r RECORD;
        BEGIN 
            FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') 
            LOOP 
                EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
            END LOOP;
        END $$;
        """
        with self.conn.cursor() as cur:
            cur.execute(query)

    def close(self):
        """Close the database connection."""
        self.conn.close()
