import uuid
from typing import List, Optional

from jet.db.postgres.client import PostgresClient
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER
from jet.llm.mlx.client import Message


class PostgresChatMessageHistory:
    """Class to manage chat message history in a PostgreSQL database using PostgresClient."""

    def __init__(
        self,
        dbname: str,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
        session_id: Optional[str] = None
    ):
        """Initialize with PostgreSQL connection parameters and session ID."""
        self.dbname = dbname
        self.session_id = session_id or str(uuid.uuid4())
        self.client = PostgresClient(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            overwrite_db=overwrite_db,
        )
        self._initialize_table()

    def _initialize_table(self):
        """Create the messages table if it doesn't exist using PostgresClient."""
        self.client._ensure_table_exists("messages")
        with self.client.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE messages
                ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) NOT NULL,
                ADD COLUMN IF NOT EXISTS role VARCHAR(50) NOT NULL,
                ADD COLUMN IF NOT EXISTS content TEXT NOT NULL,
                ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
            """)
            self.client.commit()

    def add_message(self, message: Message):
        """Add a message to the database using PostgresClient."""
        row_data = {
            "session_id": self.session_id,
            "role": message["role"],
            "content": message["content"]
        }
        self.client.create_row("messages", row_data)

    def get_messages(self) -> List[Message]:
        """Retrieve all messages for the session using PostgresClient."""
        rows = self.client.get_rows(
            table_name="messages",
            ids=None
        )
        return [
            {"role": row["role"], "content": row["content"]}
            for row in rows
            if row["session_id"] == self.session_id
        ]

    def clear(self):
        """Clear all messages for the session using PostgresClient."""
        rows = self.client.get_rows("messages")
        session_rows = [
            row for row in rows if row["session_id"] == self.session_id]
        if session_rows:
            with self.client.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM messages
                    WHERE session_id = %s;
                """, (self.session_id,))
                self.client.commit()

    def __del__(self):
        """Ensure the database connection is closed."""
        if hasattr(self, 'client') and self.client:
            self.client.close()


class ChatHistory:
    """Class to manage chat history, with optional PostgreSQL persistence."""

    def __init__(
        self,
        dbname: Optional[str] = None,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
        session_id: Optional[str] = None
    ):
        """Initialize with optional PostgreSQL connection parameters and session ID."""
        self.session_id = session_id or str(uuid.uuid4())
        self.use_db = dbname is not None

        if self.use_db and dbname:
            self.db_history = PostgresChatMessageHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                overwrite_db=overwrite_db,
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
