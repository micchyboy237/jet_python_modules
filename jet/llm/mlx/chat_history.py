import uuid
from typing import List, Optional, Union
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, TypedDict
from jet.data.utils import generate_hash, generate_unique_hash
from jet.db.postgres.client import PostgresClient
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER
from jet.llm.mlx.client import Message
from jet.models.model_types import ChatRole
import logging

logger = logging.getLogger(__name__)

ChatRole = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict):
    role: ChatRole
    content: str


class DBMessage(Message):
    id: str
    session_id: str
    message_order: int
    updated_at: Optional[datetime]
    created_at: Optional[datetime]


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
        """Ensure the messages table has the correct schema using PostgresClient."""
        logger.debug("Initializing messages table")
        # Define the expected schema for the messages table
        schema_template: DBMessage = {
            "id": str(uuid.uuid4()),  # VARCHAR(255) PRIMARY KEY
            "session_id": self.session_id,  # VARCHAR(255) NOT NULL
            "role": "system",  # VARCHAR(50) NOT NULL
            "content": "example",  # TEXT NOT NULL
            "message_order": 1,  # INTEGER NOT NULL
            "updated_at": None,  # TIMESTAMP
            "created_at": None,  # TIMESTAMP
        }
        # Ensure table exists and has all required columns
        self.client._ensure_table_exists("messages")
        self.client._ensure_columns_exist("messages", schema_template)
        # Ensure constraints (e.g., NOT NULL) are applied
        with self.client.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE messages
                ALTER COLUMN session_id SET NOT NULL,
                ALTER COLUMN role SET NOT NULL,
                ALTER COLUMN content SET NOT NULL,
                ALTER COLUMN message_order SET NOT NULL;
            """)
            # Ensure id is the primary key
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conrelid = 'messages'::regclass
                        AND contype = 'p'
                    ) THEN
                        ALTER TABLE messages ADD PRIMARY KEY (id);
                    END IF;
                END $$;
            """)
            self.client.commit()
        logger.debug("Messages table initialized successfully")

    def add_message(self, message: Message) -> DBMessage:
        """Add a message to the database using PostgresClient."""
        logger.debug(f"Adding message for session_id: {self.session_id}")
        with self.client.conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(message_order), 0) AS coalesce
                FROM messages
                WHERE session_id = %s;
            """, (self.session_id,))
            result = cur.fetchone()
            logger.debug(f"Query result for max message_order: {result}")
            max_order = result['coalesce'] if result else 0
            next_order = max_order + 1
            logger.debug(f"Calculated next_order: {next_order}")
        # Use generate_hash for id if not provided
        id_val = generate_hash(
            f"{self.session_id}-{message.get('role', '')}-{message.get('content', '')}-{next_order}"
        )
        now = datetime.now(ZoneInfo("America/Los_Angeles"))  # PST timezone
        row_data: DBMessage = {
            "id": id_val,
            "session_id": self.session_id,
            "role": message.get("role"),
            "content": message.get("content"),
            "message_order": next_order,
            "updated_at": now,
            "created_at": now,
        }
        logger.debug(f"Row data to insert: {row_data}")
        self.client.create_or_update_row("messages", row_data)
        return row_data

    def add_messages(self, messages: List[Message]) -> List[DBMessage]:
        """Add multiple messages to the database using create_or_update_rows."""
        logger.debug(
            f"Adding {len(messages)} messages for session_id: {self.session_id}")
        if not messages:
            logger.debug("No messages to add")
            return []

        db_messages: List[DBMessage] = []
        with self.client.conn.cursor() as cur:
            # Get the current max message_order
            cur.execute("""
                SELECT COALESCE(MAX(message_order), 0) AS coalesce
                FROM messages
                WHERE session_id = %s;
            """, (self.session_id,))
            result = cur.fetchone()
            max_order = result['coalesce'] if result else 0
            now = datetime.now(ZoneInfo("America/Los_Angeles"))  # PST timezone

            for idx, message in enumerate(messages, start=1):
                next_order = max_order + idx
                id_val = generate_hash(
                    f"{self.session_id}-{message.get('role', '')}-{message.get('content', '')}-{next_order}"
                )
                row_data: DBMessage = {
                    "id": id_val,
                    "session_id": self.session_id,
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "message_order": next_order,
                    "updated_at": now,
                    "created_at": now,
                }
                db_messages.append(row_data)

        # Use create_or_update_rows for bulk insert/update
        self.client.create_or_update_rows("messages", db_messages)
        logger.debug(f"Added/updated {len(db_messages)} messages")
        return db_messages

    def get_messages(self) -> List[DBMessage]:
        """Retrieve all messages for the session using PostgresClient, ordered by message_order."""
        logger.debug(f"Retrieving messages for session_id: {self.session_id}")
        try:
            rows: List[DBMessage] = self.client.get_rows(
                table_name="messages",
                where_conditions={
                    "session_id": self.session_id
                },
                order_by=("message_order", "ASC")
            )
            messages = [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "message_order": row["message_order"],
                    "role": row["role"],
                    "content": row["content"],
                    "updated_at": row["updated_at"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            logger.debug(f"Retrieved messages: {messages}")
            return messages
        except ValueError as e:
            if "Column session_id not found" in str(e):
                logger.warning(
                    "Schema mismatch detected, reinitializing table")
                self._initialize_table()
                return []
            raise

    def clear(self):
        """Clear all messages for the session using PostgresClient."""
        logger.debug(f"Clearing messages for session_id: {self.session_id}")
        rows: List[DBMessage] = self.client.get_rows("messages")
        session_rows: List[DBMessage] = [
            row for row in rows if row["session_id"] == self.session_id]
        if session_rows:
            with self.client.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM messages
                    WHERE session_id = %s;
                """, (self.session_id,))
                self.client.commit()
            logger.debug(
                f"Deleted {len(session_rows)} messages for session_id: {self.session_id}")

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
        self.messages: Union[List[DBMessage], List[Message]] = []
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
            self.messages = self.db_history.get_messages()
        else:
            self.db_history = None
            self.messages = []

    def add_message(self, role: ChatRole, content: str, id: Optional[str] = None):
        """Add a message to the history and persist it if using database."""
        if self.use_db and self.db_history:
            message: Message = {"role": role, "content": content}
            result = self.db_history.add_message(message)
            self.messages.append(result)
        else:
            now = datetime.now(ZoneInfo("America/Los_Angeles"))  # PST timezone
            message: DBMessage = {
                "role": role,
                "content": content,
                "id": id or generate_hash(f"{self.session_id}-{role}-{content}-{len(self.messages) + 1}"),
                "session_id": self.session_id,
                "message_order": len(self.messages) + 1,
                "updated_at": now,
                "created_at": now,
            }
            self.messages.append(message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to the history, updating rows with same session_id and message_order or creating new ones."""
        logger.debug("Adding %d messages for session_id: %s",
                     len(messages), self.session_id)
        if not messages:
            logger.debug("No messages to add")
            return

        # Get current messages to check for duplicates
        current_messages = self.get_messages()
        logger.debug("Current messages: %s", current_messages)
        # Filter out duplicates based on role and content only (allow same message_order for updates)
        existing_keys = {(m["role"], m["content"]) for m in current_messages}
        new_messages = []
        for message in messages:
            if (message["role"], message["content"]) not in existing_keys:
                new_messages.append(message)
            else:
                logger.debug("Skipping duplicate message: role=%s, content=%s",
                             message["role"], message["content"])

        if not new_messages:
            logger.debug("No new messages after filtering duplicates")
            return

        if self.use_db and self.db_history:
            # Persist to database, relying on create_or_update_rows to update or create rows
            db_messages = self.db_history.add_messages(new_messages)
            logger.debug("Persisted %d messages to database", len(db_messages))
            # Update in-memory messages by removing old entries with same message_order
            max_order = max((m["message_order"]
                            for m in current_messages), default=0)
            for db_message in db_messages:
                self.messages = [
                    m for m in self.messages if m["message_order"] != db_message["message_order"]]
                self.messages.append(db_message)
        else:
            # Add to in-memory messages with calculated message_order and timestamps
            now = datetime.now(ZoneInfo("America/Los_Angeles"))  # PST timezone
            max_order = max((m["message_order"]
                            for m in current_messages), default=0)
            for idx, message in enumerate(new_messages, start=1):
                next_order = max_order + idx
                db_message: DBMessage = {
                    "id": generate_hash(f"{self.session_id}-{message['role']}-{message['content']}-{next_order}"),
                    "session_id": self.session_id,
                    "role": message["role"],
                    "content": message["content"],
                    "message_order": next_order,
                    "updated_at": now,
                    "created_at": now,
                }
                logger.debug("Adding in-memory message: %s", db_message)
                # Replace existing message with same message_order, if any
                self.messages = [
                    m for m in self.messages if m["message_order"] != next_order]
                self.messages.append(db_message)
        logger.debug("Added %d messages to history", len(new_messages))

    def get_messages(self) -> List[DBMessage]:
        """Return the current list of messages."""
        if self.use_db and self.db_history:
            return self.db_history.get_messages()
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
