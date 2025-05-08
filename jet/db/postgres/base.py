import psycopg
from psycopg.rows import dict_row
from typing import Optional
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


class PostgresDB:
    """A utility class for managing PostgreSQL database connections and operations."""

    def __init__(
        self,
        default_db: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """Initialize PostgresDB with connection parameters.

        Args:
            default_db (str): Name of the default database.
            user (str): Database user.
            password (str): Database password.
            host (str): Database host address.
            port (int): Database port number.
        """
        self.default_db = default_db
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connect_db(
        self,
        dbname: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> psycopg.Connection:
        """Establish a connection to a specified database.

        Args:
            dbname (str): Name of the database to connect to.
            user (Optional[str]): Database user. Defaults to instance user.
            password (Optional[str]): Database password. Defaults to instance password.
            host (Optional[str]): Database host. Defaults to instance host.
            port (Optional[int]): Database port. Defaults to instance port.

        Returns:
            psycopg.Connection: A connection object to the specified database.

        Raises:
            Exception: If the connection attempt fails.
        """
        try:
            return psycopg.connect(
                dbname=dbname,
                user=user or self.user,
                password=password or self.password,
                host=host or self.host,
                port=port or self.port,
                autocommit=True,
                row_factory=dict_row,
            )
        except psycopg.Error as e:
            raise Exception(
                f"Failed to connect to database {dbname}: {str(e)}")

    def connect_default_db(self) -> psycopg.Connection:
        """Connect to the default database, creating it if it doesn't exist.

        Returns:
            psycopg.Connection: A connection object to the default database.

        Raises:
            Exception: If database creation or connection fails.
        """
        # Check if default database exists by connecting to 'postgres'
        try:
            with self.connect_db(
                dbname="postgres",
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s", (
                            self.default_db,)
                    )
                    if not cur.fetchone():
                        cur.execute(f"CREATE DATABASE {self.default_db}")
        except psycopg.Error as e:
            raise Exception(
                f"Failed to verify or create database {self.default_db}: {str(e)}")

        # Connect to the default database
        return self.connect_db(
            dbname=self.default_db,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
        )

    def create_db(self, dbname: str) -> None:
        """Create a new database.

        Args:
            dbname (str): Name of the database to create.

        Raises:
            Exception: If database creation fails.
        """
        with self.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"CREATE DATABASE {dbname}")
                except psycopg.Error as e:
                    raise Exception(
                        f"Failed to create database {dbname}: {str(e)}")

    def delete_db(self, dbname: str) -> None:
        """Delete a specified database.

        Args:
            dbname (str): Name of the database to delete.

        Raises:
            Exception: If database deletion fails.
        """
        with self.connect_default_db() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"DROP DATABASE IF EXISTS {dbname}")
                except psycopg.Error as e:
                    raise Exception(
                        f"Failed to delete database {dbname}: {str(e)}")


__all__ = ["PostgresDB"]
