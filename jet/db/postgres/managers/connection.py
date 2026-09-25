from pgvector.psycopg import register_vector
from psycopg import connect, sql
from psycopg.rows import dict_row

from jet.logger import logger


class ConnectionManager:
    """Manages database connections and lifecycle."""

    def __init__(
        self,
        dbname: str,
        user: str,
        password: str,
        host: str,
        port: int,
        overwrite_db: bool = False,
    ):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self._ensure_database_exists(overwrite_db)
        self._connect()

    def _connect(self):
        self.conn = connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            autocommit=True,
            row_factory=dict_row,
        )

    def _ensure_database_exists(self, overwrite_db: bool):
        with connect(
            dbname="postgres",
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            autocommit=True,
        ) as admin_conn:
            with admin_conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;", (self.dbname,)
                )
                exists = cur.fetchone()
                if exists and overwrite_db:
                    logger.info(f"Overwriting database: {self.dbname}")
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (self.dbname,),
                    )
                    cur.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {}").format(
                            sql.Identifier(self.dbname)
                        )
                    )
                if not exists or overwrite_db:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(self.dbname)
                        )
                    )
                    logger.info(f"Created database: {self.dbname}")

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()

    def delete_db(self):
        self.close()
        try:
            with connect(
                dbname="postgres",
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                autocommit=True,
            ) as admin_conn:
                with admin_conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (self.dbname,),
                    )
                    cur.execute(
                        sql.SQL("DROP DATABASE IF EXISTS {}").format(
                            sql.Identifier(self.dbname)
                        )
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete database {self.dbname}: {str(e)}"
            ) from e

    def begin_transaction(self) -> None:
        """Explicitly begin a new transaction."""
        if self.conn:
            self.conn.execute("BEGIN;")

    def commit(self) -> None:
        """Commit the current transaction."""
        if self.conn:
            self.conn.execute("COMMIT;")

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self.conn:
            self.conn.execute("ROLLBACK;")

    def __enter__(self):
        if self.conn:
            self.conn.execute("BEGIN;")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn and not self.conn.closed:
            if exc_type is None:
                self.conn.execute("COMMIT;")
            else:
                self.conn.execute("ROLLBACK;")
