import pytest
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List
from jet.llm.mlx.chat_history import PostgresChatMessageHistory, ChatHistory, Message, DBMessage
from jet.db.postgres.client import PostgresClient


@pytest.fixture
def db_client():
    """Fixture to provide a PostgresClient instance."""
    client = PostgresClient(
        dbname="test_db",
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        overwrite_db=True
    )
    yield client
    client.close()


@pytest.fixture
def chat_history(db_client):
    """Fixture to provide a ChatHistory instance with database."""
    history = ChatHistory(
        dbname="test_db",
        user="test_user",
        password="test_password",
        host="localhost",
        port=5432,
        overwrite_db=True,
        session_id=str(uuid.uuid4()),
        conversation_id=str(uuid.uuid4())
    )
    yield history
    history.clear()


class TestPostgresChatMessageHistory:
    def test_add_and_retrieve_messages_by_conversation_id(self, db_client):
        # Given: A PostgresChatMessageHistory instance with a specific conversation_id
        session_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        history = PostgresChatMessageHistory(
            dbname="test_db",
            user="test_user",
            password="test_password",
            host="localhost",
            port=5432,
            overwrite_db=True,
            session_id=session_id,
            conversation_id=conversation_id
        )

        # When: Adding messages to the same conversation
        message1: Message = {"role": "user", "content": "Hello"}
        message2: Message = {"role": "assistant", "content": "Hi there"}
        expected_message1 = history.add_message(message1)
        expected_message2 = history.add_message(message2)

        # Then: Messages are retrieved in correct order by conversation_id
        result = history.get_messages()
        expected = [
            {
                "id": expected_message1["id"],
                "session_id": session_id,
                "conversation_id": conversation_id,
                "role": "user",
                "content": "Hello",
                "message_order": 1,
                "updated_at": expected_message1["updated_at"],
                "created_at": expected_message1["created_at"]
            },
            {
                "id": expected_message2["id"],
                "session_id": session_id,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": "Hi there",
                "message_order": 2,
                "updated_at": expected_message2["updated_at"],
                "created_at": expected_message2["created_at"]
            }
        ]
        assert result == expected, f"Expected messages {expected}, but got {result}"

    def test_clear_messages_by_conversation_id(self, db_client):
        # Given: A PostgresChatMessageHistory with messages in a conversation
        session_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        history = PostgresChatMessageHistory(
            dbname="test_db",
            user="test_user",
            password="test_password",
            host="localhost",
            port=5432,
            overwrite_db=True,
            session_id=session_id,
            conversation_id=conversation_id
        )
        message: Message = {"role": "user", "content": "Test message"}
        history.add_message(message)

        # When: Clearing the conversation
        history.clear()

        # Then: No messages are retrieved for the conversation_id
        result = history.get_messages()
        expected: List[DBMessage] = []
        assert result == expected, f"Expected empty message list, but got {result}"


class TestChatHistory:
    def test_add_and_retrieve_messages_in_memory(self):
        # Given: An in-memory ChatHistory with a specific conversation_id
        session_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        history = ChatHistory(session_id=session_id,
                              conversation_id=conversation_id)

        # When: Adding messages
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")

        # Then: Messages are retrieved with correct conversation_id and order
        result = history.get_messages()
        now = datetime.now(ZoneInfo("America/Los_Angeles"))
        expected = [
            {
                "id": result[0]["id"],
                "session_id": session_id,
                "conversation_id": conversation_id,
                "role": "user",
                "content": "Hello",
                "message_order": 1,
                "updated_at": result[0]["updated_at"],
                "created_at": result[0]["created_at"]
            },
            {
                "id": result[1]["id"],
                "session_id": session_id,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": "Hi there",
                "message_order": 2,
                "updated_at": result[1]["updated_at"],
                "created_at": result[1]["created_at"]
            }
        ]
        assert len(result) == 2, f"Expected 2 messages, got {len(result)}"
        assert result[0]["conversation_id"] == conversation_id, "Incorrect conversation_id"
        assert result[0]["message_order"] == 1, "Incorrect message order for first message"
        assert result[1]["message_order"] == 2, "Incorrect message order for second message"

    def test_add_multiple_messages_with_conversation_id(self, chat_history):
        # Given: A ChatHistory with database and a specific conversation_id
        conversation_id = chat_history.conversation_id
        messages: List[Message] = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response to first"}
        ]

        # When: Adding multiple messages
        chat_history.add_messages(messages)

        # Then: Messages are stored and retrieved correctly
        result = chat_history.get_messages()
        expected = [
            {
                "id": result[0]["id"],
                "session_id": chat_history.session_id,
                "conversation_id": conversation_id,
                "role": "user",
                "content": "First message",
                "message_order": 1,
                "updated_at": result[0]["updated_at"],
                "created_at": result[0]["created_at"]
            },
            {
                "id": result[1]["id"],
                "session_id": chat_history.session_id,
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": "Response to first",
                "message_order": 2,
                "updated_at": result[1]["updated_at"],
                "created_at": result[1]["created_at"]
            }
        ]
        assert result == expected, f"Expected messages {expected}, but got {result}"
