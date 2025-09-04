import pytest
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from jet.llm.mlx.remote.client import MLXRemoteClient
from jet.llm.mlx.remote.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    HealthResponse,
    ModelsResponse
)


class MockServerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        elif self.path.startswith("/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "object": "list",
                "data": [{"id": "test-model", "object": "model", "created": 1234567890}]
            }
            self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        body = json.loads(raw_body.decode())

        if self.path in ["/v1/chat/completions", "/chat/completions"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if body.get("stream", False):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1234567890,
                    "model": body.get("model", "test-model"),
                    "system_fingerprint": "test-fingerprint",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.write("data: [DONE]\n\n".encode())
            else:
                self.end_headers()
                response = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": body.get("model", "test-model"),
                    "system_fingerprint": "test-fingerprint",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
                self.wfile.write(json.dumps(response).encode())
        elif self.path == "/v1/completions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if body.get("stream", False):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                response = {
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "created": 1234567890,
                    "model": body.get("model", "test-model"),
                    "system_fingerprint": "test-fingerprint",
                    "choices": [{"index": 0, "text": "Hello", "finish_reason": None}]
                }
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.write("data: [DONE]\n\n".encode())
            else:
                self.end_headers()
                response = {
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "created": 1234567890,
                    "model": body.get("model", "test-model"),
                    "system_fingerprint": "test-fingerprint",
                    "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                }
                self.wfile.write(json.dumps(response).encode())


@pytest.fixture
def mock_server():
    server = HTTPServer(("localhost", 8081), MockServerHandler)
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    yield server
    server.shutdown()
    server.server_close()


class TestMLXRemoteClient:
    @pytest.fixture(autouse=True)
    def setup(self, mock_server):
        self.client = MLXRemoteClient("http://localhost:8081")
        yield

    def test_health_check(self):
        # Given: A running MLX server
        # When: Checking the health endpoint
        result = self.client.health_check()
        # Then: Should return status ok
        expected: HealthResponse = {"status": "ok"}
        assert result == expected

    def test_list_models(self):
        # Given: A running MLX server with available models
        # When: Listing available models
        result = self.client.list_models()
        # Then: Should return a list of models
        expected: ModelsResponse = {
            "object": "list",
            "data": [{"id": "test-model", "object": "model", "created": 1234567890}]
        }
        assert result == expected

    def test_create_chat_completion(self):
        # Given: A valid chat completion request
        request: ChatCompletionRequest = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model"
        }
        # When: Creating a chat completion
        result = self.client.create_chat_completion(request)
        # Then: Should return a valid chat completion response
        expected: ChatCompletionResponse = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "system_fingerprint": "test-fingerprint",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        assert result == expected

    def test_create_chat_completion_stream(self):
        # Given: A valid chat completion request with streaming
        request: ChatCompletionRequest = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "stream": True
        }
        # When: Creating a streaming chat completion
        result = self.client.create_chat_completion(request, stream=True)
        # Then: Should return a list of chat completion chunks
        expected = [{
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "system_fingerprint": "test-fingerprint",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]
        }]
        assert result == expected

    def test_create_text_completion(self):
        # Given: A valid text completion request
        request: TextCompletionRequest = {
            "prompt": "Hello",
            "model": "test-model"
        }
        # When: Creating a text completion
        result = self.client.create_text_completion(request)
        # Then: Should return a valid text completion response
        expected: TextCompletionResponse = {
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 1234567890,
            "model": "test-model",
            "system_fingerprint": "test-fingerprint",
            "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        assert result == expected

    def test_create_text_completion_stream(self):
        # Given: A valid text completion request with streaming
        request: TextCompletionRequest = {
            "prompt": "Hello",
            "model": "test-model",
            "stream": True
        }
        # When: Creating a streaming text completion
        result = self.client.create_text_completion(request, stream=True)
        # Then: Should return a list of text completion chunks
        expected = [{
            "id": "cmpl-test",
            "object": "text_completion",
            "created": 1234567890,
            "model": "test-model",
            "system_fingerprint": "test-fingerprint",
            "choices": [{"index": 0, "text": "Hello", "finish_reason": None}]
        }]
        assert result == expected
