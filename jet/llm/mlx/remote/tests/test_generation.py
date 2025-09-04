# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/mlx/remote/tests/test_generation.py
import pytest
from unittest.mock import MagicMock, patch

import jet.llm.mlx.remote.generation as gen
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.remote.types import Message


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.list_models.return_value = {"models": ["mlx-test"]}
    client.health_check.return_value = {"status": "ok"}
    client.create_chat_completion.return_value = {
        "id": "chat-123", "object": "chat.completion"}
    client.create_text_completion.return_value = {
        "id": "text-123", "object": "text_completion"}
    return client


def test_get_models_with_client(mock_client):
    resp = gen.get_models(client=mock_client, repo_id="test-repo")
    assert resp["models"] == ["mlx-test"]
    mock_client.list_models.assert_called_once_with(repo_id="test-repo")


def test_get_models_without_client(monkeypatch, mock_client):
    monkeypatch.setattr(gen, "MLXRemoteClient", lambda base_url: mock_client)
    resp = gen.get_models(base_url="http://fake")
    assert "models" in resp


def test_health_check_with_client(mock_client):
    resp = gen.health_check(client=mock_client)
    assert resp["status"] == "ok"
    mock_client.health_check.assert_called_once()


def test_health_check_without_client(monkeypatch, mock_client):
    monkeypatch.setattr(gen, "MLXRemoteClient", lambda base_url: mock_client)
    resp = gen.health_check(base_url="http://fake")
    assert resp["status"] == "ok"


def test_prepare_messages_str_with_history():
    history = ChatHistory()
    msgs = gen.prepare_messages(
        "Hello", history, system_prompt="sys", with_history=True)
    assert any(m["role"] == "system" for m in msgs)
    assert any(m["role"] == "user" and m["content"] == "Hello" for m in msgs)


def test_prepare_messages_list_without_history():
    messages = [{"role": "user", "content": "Hi"}]
    history = ChatHistory()
    msgs = gen.prepare_messages(
        messages, history, system_prompt="sys", with_history=False)
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_prepare_messages_invalid_type():
    history = ChatHistory()
    with pytest.raises(TypeError):
        gen.prepare_messages(123, history)  # type: ignore[arg-type]


def test_prepare_messages_missing_keys():
    history = ChatHistory()
    with pytest.raises(ValueError):
        gen.prepare_messages([{"role": "user"}], history)


def test_chat_calls_client(mock_client):
    resp = gen.chat("Hello", client=mock_client, model="mlx-test")
    assert resp["id"] == "chat-123"
    mock_client.create_chat_completion.assert_called_once()
    args, kwargs = mock_client.create_chat_completion.call_args
    assert args[0]["messages"][0]["content"] == "Hello"


def test_stream_chat_yields_chunks(mock_client):
    mock_client.create_chat_completion.return_value = [
        {"id": "c1"}, {"id": "c2"}
    ]
    chunks = list(gen.stream_chat("Hi", client=mock_client))
    assert len(chunks) == 2
    assert chunks[0]["id"] == "c1"


def test_generate_calls_client(mock_client):
    resp = gen.generate("Once upon a time",
                        client=mock_client, model="mlx-test")
    assert resp["id"] == "text-123"
    mock_client.create_text_completion.assert_called_once()


def test_stream_generate_yields_chunks(mock_client):
    mock_client.create_text_completion.return_value = [
        {"id": "t1"}, {"id": "t2"}
    ]
    chunks = list(gen.stream_generate("Test", client=mock_client))
    assert len(chunks) == 2
    assert chunks[1]["id"] == "t2"
