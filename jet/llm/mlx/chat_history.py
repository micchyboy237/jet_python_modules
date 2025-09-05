import uuid
from typing import List, Optional, Union
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal, TypedDict, Dict
from jet.data.utils import generate_hash
from jet.db.postgres.client import PostgresClient
from jet.db.postgres.config import DEFAULT_HOST, DEFAULT_PASSWORD, DEFAULT_PORT, DEFAULT_USER
from jet.models.model_types import ChatRole
import logging
import json

logger = logging.getLogger(__name__)

ChatRole = Literal["system", "user", "assistant", "tool"]


class Message(TypedDict):
    role: ChatRole
    content: str
    tool_calls: Optional[List[Dict]]


class DBMessage(Message):
    id: str
    session_id: str
    conversation_id: str
    message_order: int
    updated_at: Optional[datetime]
    created_at: Optional[datetime]
    name: Optional[str]


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
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        name: Optional[str] = None
    ):
        """Initialize with PostgreSQL connection parameters, session ID, conversation ID, and optional name."""
        self.dbname = dbname
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.name = name
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
        schema_template: DBMessage = {
            "id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "role": "system",
            "content": "example",
            "tool_calls": None,
            "message_order": 1,
            "updated_at": None,
            "created_at": None,
            "name": None,
        }
        self.client._ensure_table_exists("messages")
        self.client._ensure_columns_exist("messages", schema_template)
        with self.client.conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE messages
                ALTER COLUMN session_id SET NOT NULL,
                ALTER COLUMN conversation_id SET NOT NULL,
                ALTER COLUMN role SET NOT NULL,
                ALTER COLUMN content SET NOT NULL,
                ALTER COLUMN message_order SET NOT NULL;
            """)
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

    def add_message(self, message: Message, name: Optional[str] = None) -> DBMessage:
        """Add a message to the database using PostgresClient."""
        logger.debug(
            f"Adding message for conversation_id: {self.conversation_id}")
        with self.client.conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(message_order), 0) AS coalesce
                FROM messages
                WHERE conversation_id = %s;
            """, (self.conversation_id,))
            result = cur.fetchone()
            logger.debug(f"Query result for max message_order: {result}")
            max_order = result['coalesce'] if result else 0
            next_order = max_order + 1
            logger.debug(f"Calculated next_order: {next_order}")
        id_val = generate_hash(
            f"{self.conversation_id}-{message.get('role', '')}-{message.get('content', '')}-{next_order}"
        )
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        tool_calls_json = json.dumps(message.get("tool_calls")) if message.get(
            "tool_calls") is not None else None
        row_data: DBMessage = {
            "id": id_val,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "role": message.get("role"),
            "content": message.get("content"),
            "tool_calls": tool_calls_json,
            "message_order": next_order,
            "updated_at": now,
            "created_at": now,
            "name": name,
        }
        logger.debug(f"Row data to insert: {row_data}")
        self.client.create_or_update_row("messages", row_data)
        return row_data

    def add_messages(self, messages: List[Message], name: Optional[str] = None) -> List[DBMessage]:
        """Add multiple messages to the database using create_or_update_rows."""
        logger.debug(
            f"Adding {len(messages)} messages for conversation_id: {self.conversation_id}")
        if not messages:
            logger.debug("No messages to add")
            return []
        db_messages: List[DBMessage] = []
        with self.client.conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(MAX(message_order), 0) AS coalesce
                FROM messages
                WHERE conversation_id = %s;
            """, (self.conversation_id,))
            result = cur.fetchone()
            max_order = result['coalesce'] if result else 0
            now = datetime.now(ZoneInfo("America/Los_Angeles"))
            for idx, message in enumerate(messages, start=1):
                next_order = max_order + idx
                id_val = generate_hash(
                    f"{self.conversation_id}-{message.get('role', '')}-{message.get('content', '')}-{next_order}"
                )
                tool_calls_json = json.dumps(message.get("tool_calls")) if message.get(
                    "tool_calls") is not None else None
                row_data: DBMessage = {
                    "id": id_val,
                    "session_id": self.session_id,
                    "conversation_id": self.conversation_id,
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "tool_calls": tool_calls_json,
                    "message_order": next_order,
                    "updated_at": now,
                    "created_at": now,
                    "name": name,
                }
                db_messages.append(row_data)
        self.client.create_or_update_rows("messages", db_messages)
        logger.debug(f"Added/updated {len(db_messages)} messages")
        return db_messages

    def get_messages(self) -> List[DBMessage]:
        """Retrieve all messages for the conversation using PostgresClient, ordered by message_order."""
        logger.debug(
            f"Retrieving messages for conversation_id: {self.conversation_id}")
        try:
            rows: List[DBMessage] = self.client.get_rows(
                table_name="messages",
                where_conditions={
                    "conversation_id": self.conversation_id
                },
                order_by=("message_order", "ASC")
            )
            messages = [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "conversation_id": row["conversation_id"],
                    "message_order": row["message_order"],
                    "role": row["role"],
                    "content": row["content"],
                    "tool_calls": json.loads(row["tool_calls"]) if row["tool_calls"] else None,
                    "updated_at": row["updated_at"],
                    "created_at": row["created_at"],
                    "name": row["name"],
                }
                for row in rows
            ]
            logger.debug(f"Retrieved messages: {messages}")
            return messages
        except ValueError as e:
            if "Column conversation_id not found" in str(e):
                logger.warning(
                    "Schema mismatch detected, reinitializing table")
                self._initialize_table()
                return []
            raise

    def clear(self):
        """Clear all messages for the conversation using PostgresClient."""
        logger.debug(
            f"Clearing messages for conversation_id: {self.conversation_id}")
        rows: List[DBMessage] = self.client.get_rows("messages")
        conversation_rows: List[DBMessage] = [
            row for row in rows if row["conversation_id"] == self.conversation_id]
        if conversation_rows:
            with self.client.conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM messages
                    WHERE conversation_id = %s;
                """, (self.conversation_id,))
                self.client.commit()
            logger.debug(
                f"Deleted {len(conversation_rows)} messages for conversation_id: {self.conversation_id}")

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
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        name: Optional[str] = None
    ):
        """Initialize with optional PostgreSQL connection parameters, session ID, conversation ID, and name."""
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.name = name
        self.use_db = dbname is not None
        self.messages: Union[List[DBMessage], List[Message]] = []
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.overwrite_db = overwrite_db
        if self.use_db and dbname:
            self.db_history = PostgresChatMessageHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                overwrite_db=overwrite_db,
                session_id=self.session_id,
                conversation_id=self.conversation_id,
                name=name
            )
            self.messages = self.db_history.get_messages()
        else:
            self.db_history = None
            self.messages = []

    def add_message(self, role: ChatRole, content: str, tool_calls: Optional[List[Dict]] = None, id: Optional[str] = None, name: Optional[str] = None):
        """Add a message to the history and persist it if using database, replacing system message if role is system."""
        if role == "system":
            # Remove existing system message if present
            self.messages = [m for m in self.messages if m["role"] != "system"]
            if self.use_db and self.db_history:
                with self.db_history.client.conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM messages
                        WHERE conversation_id = %s AND role = %s;
                    """, (self.conversation_id, "system"))
                    self.db_history.client.commit()
        if self.use_db and self.db_history:
            message: Message = {"role": role,
                                "content": content, "tool_calls": tool_calls}
            result = self.db_history.add_message(message, name=name)
            self.messages = [m for m in self.messages if m["role"]
                             != "system" or role != "system"]
            self.messages.append(result)
        else:
            now = datetime.now(ZoneInfo("America/Los_Angeles"))
            max_order = max((m["message_order"]
                            for m in self.messages), default=0)
            next_order = max_order + 1
            message: DBMessage = {
                "id": id or generate_hash(f"{self.conversation_id}-{role}-{content}-{next_order}"),
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "message_order": next_order,
                "updated_at": now,
                "created_at": now,
                "name": name,
            }
            self.messages = [m for m in self.messages if m["role"]
                             != "system" or role != "system"]
            self.messages.append(message)

    def add_messages(self, messages: List[Message], name: Optional[str] = None) -> None:
        """Add multiple messages to the history, replacing system message if present, updating rows with same conversation_id and message_order or creating new ones."""
        logger.debug("Adding %d messages for conversation_id: %s",
                     len(messages), self.conversation_id)
        if not messages:
            logger.debug("No messages to add")
            return
        current_messages = self.get_messages()
        logger.debug("Current messages: %s", current_messages)
        existing_keys = {(m["role"], m["content"], str(
            m.get("tool_calls"))) for m in current_messages}
        new_messages = []
        has_system_message = any(m["role"] == "system" for m in messages)
        if has_system_message:
            # Remove existing system message from in-memory and database
            self.messages = [m for m in self.messages if m["role"] != "system"]
            if self.use_db and self.db_history:
                with self.db_history.client.conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM messages
                        WHERE conversation_id = %s AND role = %s;
                    """, (self.conversation_id, "system"))
                    self.db_history.client.commit()
        for message in messages:
            message_key = (message["role"], message["content"], str(
                message.get("tool_calls")))
            if message_key not in existing_keys:
                new_messages.append(message)
            else:
                logger.debug("Skipping duplicate message: role=%s, content=%s, tool_calls=%s",
                             message["role"], message["content"], message.get("tool_calls"))
        if not new_messages:
            logger.debug("No new messages after filtering duplicates")
            return
        if self.use_db and self.db_history:
            db_messages = self.db_history.add_messages(new_messages, name=name)
            logger.debug("Persisted %d messages to database", len(db_messages))
            max_order = max(
                (m["message_order"] for m in current_messages if m["role"] != "system"), default=0)
            for db_message in db_messages:
                self.messages = [m for m in self.messages if m["message_order"]
                                 != db_message["message_order"] and m["role"] != "system"]
                self.messages.append(db_message)
        else:
            now = datetime.now(ZoneInfo("America/Los_Angeles"))
            max_order = max(
                (m["message_order"] for m in current_messages if m["role"] != "system"), default=0)
            for idx, message in enumerate(new_messages, start=1):
                next_order = max_order + idx
                db_message: DBMessage = {
                    "id": generate_hash(f"{self.conversation_id}-{message['role']}-{message['content']}-{next_order}"),
                    "session_id": self.session_id,
                    "conversation_id": self.conversation_id,
                    "role": message["role"],
                    "content": message["content"],
                    "tool_calls": message.get("tool_calls"),
                    "message_order": next_order,
                    "updated_at": now,
                    "created_at": now,
                    "name": name,
                }
                logger.debug("Adding in-memory message: %s", db_message)
                self.messages = [
                    m for m in self.messages if m["message_order"] != next_order and m["role"] != "system"]
                self.messages.append(db_message)
            logger.debug("Added %d messages to in-memory history",
                         len(new_messages))

    def get_messages(self) -> List[DBMessage]:
        """Return the current list of messages, with system message at the top if present."""
        messages = self.db_history.get_messages(
        ) if self.use_db and self.db_history else self.messages
        # Sort messages to ensure system message is at the top
        system_messages = [m for m in messages if m["role"] == "system"]
        other_messages = sorted(
            [m for m in messages if m["role"] != "system"], key=lambda x: x["message_order"])
        return system_messages + other_messages

    def clear(self):
        """Clear the in-memory and persistent history if using database."""
        self.messages = []
        if self.use_db and self.db_history:
            self.db_history.clear()


__all__ = [
    "PostgresChatMessageHistory",
    "ChatHistory",
]
