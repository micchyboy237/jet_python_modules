import json
import requests
from typing import Dict, List, Optional, Union, Literal
from jet.llm.mlx.remote.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    ModelsResponse,
    HealthResponse
)


class MLXRemoteClient:
    def __init__(self, base_url: Optional[str] = None):
        """Initialize the MLX remote client with the server base URL."""
        if base_url is None:
            base_url = "http://jethros-macbook-air.local:8080"
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> HealthResponse:
        """Check the health status of the MLX server."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_models(self, repo_id: Optional[str] = None) -> ModelsResponse:
        """List available models, optionally filtered by repo_id."""
        url = f"{self.base_url}/v1/models"
        if repo_id:
            url += f"/{repo_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        stream: bool = False
    ) -> Union[ChatCompletionResponse, List[ChatCompletionResponse]]:
        """Create a chat completion with the given request parameters."""
        headers = {"Content-Type": "application/json"}
        request_dict = request.copy()
        request_dict["stream"] = stream
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_dict,
            headers=headers,
            stream=stream
        )
        response.raise_for_status()

        if stream:
            result = []
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        if decoded == "data: [DONE]":
                            break
                        result.append(json.loads(decoded[6:]))
            return result
        return response.json()

    def create_text_completion(
        self,
        request: TextCompletionRequest,
        stream: bool = False
    ) -> Union[TextCompletionResponse, List[TextCompletionResponse]]:
        """Create a text completion with the given request parameters."""
        headers = {"Content-Type": "application/json"}
        request_dict = request.copy()
        request_dict["stream"] = stream
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=request_dict,
            headers=headers,
            stream=stream
        )
        response.raise_for_status()

        if stream:
            result = []
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        if decoded == "data: [DONE]":
                            break
                        result.append(json.loads(decoded[6:]))
            return result
        return response.json()
