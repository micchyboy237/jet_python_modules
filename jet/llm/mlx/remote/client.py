import logging
from typing import Dict, Optional, Union, List, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

from jet.llm.mlx.remote.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    ModelsResponse,
    HealthResponse,
)
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.object import remove_null_keys


MLX_REMOTE_URL = "http://jethros-macbook-air.local:8080"


class MLXRemoteClient:
    """Client for interacting with the MLX server API."""

    def __init__(self, base_url: Optional[str] = None, verbose: bool = False):
        """Initialize the MLX client with a base URL and verbose logging option."""
        self.base_url = base_url or MLX_REMOTE_URL
        self.verbose = verbose
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def list_models(self, repo_id: Optional[str] = None) -> ModelsResponse:
        """List available models from the MLX server."""
        url = f"{self.base_url}/v1/models"
        params = {"repo_id": repo_id} if repo_id else None
        if self.verbose:
            logger.info(f"Requesting models from {url} with params: {params}")
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            if self.verbose:
                logger.success(
                    f"Models response:\n{format_json(response.json())}")
            return response.json()
        except requests.RequestException as e:
            if self.verbose:
                logger.error(f"Failed to list models: {e}")
            raise

    def health_check(self) -> HealthResponse:
        """Perform a health check on the MLX server."""
        url = f"{self.base_url}/health"
        if self.verbose:
            logger.info(f"Checking health at {url}")
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            if self.verbose:
                logger.success(f"Health check response: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            if self.verbose:
                logger.error(f"Health check failed: {e}")
            raise

    def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        stream: bool = False
    ) -> Union[ChatCompletionResponse, Iterator[ChatCompletionResponse]]:
        """Create a chat completion via the MLX server."""
        url = f"{self.base_url}/v1/chat/completions"
        # Remove None values from the request
        cleaned_request = remove_null_keys(request)
        if self.verbose:
            logger.info(f"Creating chat completion at {url} with request:")
            logger.debug(format_json(cleaned_request))
        try:
            response = self.session.post(
                url,
                json=cleaned_request,
                stream=stream,
                timeout=(10, 30)
            )
            response.raise_for_status()
            if stream:
                for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if line:
                        line = line.strip()
                        if line.startswith("data: "):
                            line = line[6:]
                        if line:
                            try:
                                chunk = json.loads(line)
                                if self.verbose:
                                    logger.teal(
                                        chunk["choices"][0]["delta"]["content"], flush=True)
                                yield chunk
                                # Check for finish_reason in choices to stop streaming
                                for choice in chunk.get("choices", []):
                                    if choice.get("finish_reason") is not None:
                                        if self.verbose:
                                            logger.success(
                                                f"Chat stream completion response:\n{format_json(chunk)}")
                                        return
                            except json.JSONDecodeError as e:
                                if self.verbose:
                                    logger.warning(
                                        f"Skipping invalid JSON chunk: {line}, error: {e}")
                                continue
            else:
                result = response.json()
                if self.verbose:
                    logger.success(
                        f"Chat completion response:\n{format_json(result)}")
                yield result
        except requests.RequestException as e:
            if self.verbose:
                logger.error(f"Chat completion failed: {e}")
            raise
        finally:
            response.close() if 'response' in locals() else None

    def create_text_completion(
        self,
        request: TextCompletionRequest,
        stream: bool = False
    ) -> Union[TextCompletionResponse, Iterator[TextCompletionResponse]]:
        """Create a text completion via the MLX server."""
        url = f"{self.base_url}/v1/completions"
        # Remove None values from the request
        cleaned_request = remove_null_keys(request)
        if self.verbose:
            logger.info(f"Creating text completion at {url} with request:")
            logger.debug(format_json(cleaned_request))
        try:
            response = self.session.post(
                url,
                json=cleaned_request,
                stream=stream,
                timeout=(10, 30)
            )
            response.raise_for_status()
            if stream:
                for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                    if line:
                        line = line.strip()
                        if line.startswith("data: "):
                            line = line[6:]
                        if line:
                            try:
                                chunk = json.loads(line)
                                if self.verbose:
                                    logger.teal(
                                        chunk["choices"][0]["text"], flush=True)
                                yield chunk
                                # Check for finish_reason in choices to stop streaming
                                for choice in chunk.get("choices", []):
                                    if choice.get("finish_reason") is not None:
                                        if self.verbose:
                                            logger.success(
                                                f"Text stream completion response:\n{format_json(chunk)}")
                                        return
                            except json.JSONDecodeError as e:
                                if self.verbose:
                                    logger.warning(
                                        f"Skipping invalid JSON chunk: {line}, error: {e}")
                                continue
            else:
                result = response.json()
                if self.verbose:
                    logger.success(
                        f"Text completion response:\n{format_json(result)}")
                yield result
        except requests.RequestException as e:
            if self.verbose:
                logger.error(f"Text completion failed: {e}")
            raise
        finally:
            response.close() if 'response' in locals() else None
