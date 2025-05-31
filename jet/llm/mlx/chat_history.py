import uuid
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row

from .client import Message


class PostgresChatMessageHistory:
    """Class to manage chat message history in a PostgreSQL database using psycopg."""

    def __init__(self, dbname: str, user: str, password: str,
                 host: str, port: str, session_id: Optional[str] = None):
        """Initialize with PostgreSQL connection parameters and session ID."""
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.session_id = session_id or str(uuid.uuid4())
        self._initialize_table()

    def _initialize_table(self):
        """Create the messages table if it doesn't exist."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()

    def add_message(self, message: Message):
        """Add a message to the database."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, role, content)
                    VALUES (%s, %s, %s);
                """, (self.session_id, message["role"], message["content"]))
                conn.commit()

    def get_messages(self) -> List[Message]:
        """Retrieve all messages for the session."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            row_factory=dict_row
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT role, content
                    FROM messages
                    WHERE session_id = %s
                    ORDER BY timestamp ASC;
                """, (self.session_id,))
                return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]

    def clear(self):
        """Clear all messages for the session."""
        with psycopg.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM messages
                    WHERE session_id = %s;
                """, (self.session_id,))
                conn.commit()


class ChatHistory:
    """Class to manage chat history, with optional PostgreSQL persistence."""

    def __init__(self, dbname: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, host: Optional[str] = None,
                 port: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize with optional PostgreSQL connection parameters and session ID."""
        self.session_id = session_id or str(uuid.uuid4())
        self.use_db = all([dbname, user, password, host, port])

        if self.use_db:
            self.db_history = PostgresChatMessageHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                session_id=self.session_id
            )
            self.messages: List[Message] = self.db_history.get_messages()
        else:
            self.db_history = None
            self.messages: List[Message] = []

    def add_message(self, role: str, content: str):
        """Add a message to the history and persist it if using database."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        if self.use_db and self.db_history:
            self.db_history.add_message(message)

    def get_messages(self) -> List[Message]:
        """Return the current list of messages."""
        return self.messages

    def clear(self):
        """Clear the in-memory and persistent history if using database."""
        self.messages = []
        if self.use_db and self.db_history:
            self.db_history.clear()


__all__ = [
    "PostgresChatMessageHistory",
    "ChatHistory",
]
