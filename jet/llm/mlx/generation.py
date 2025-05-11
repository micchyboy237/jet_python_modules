# jet_python_modules/jet/llm/mlx/generation.py
import json
import time
import uuid
from jet.llm.mlx.mlx_types import ModelKey
from jet.llm.mlx.models import resolve_model
from jet.transformers.formatters import format_json
import requests
from typing import List, Dict, Optional, Union, Literal, Generator
from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException
from jet.logger import logger
from jet.llm.mlx.model_cache import MODEL_CACHE, MODEL_LIST_CACHE_LOCK
from jet.llm.mlx.mlx_class_types import (
    Message,
    Usage,
    UnifiedCompletionResponse,
    ModelsResponse,
    ParallelCompletionResponse,
)

BASE_URL = "http://localhost:9000"


def _handle_response(response: requests.Response, is_stream: bool) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response headers: {response.headers}")
    content_type = response.headers.get("Content-Type", "")
    expected_content_type = "application/json"
    if expected_content_type not in content_type:
        logger.error(
            f"Unexpected Content-Type: {content_type}, expected {expected_content_type}")
        raise HTTPException(
            status_code=500, detail=f"Server returned unexpected Content-Type: {content_type}")
    response.raise_for_status()

    def estimate_tokens(text: Optional[str]) -> int:
        """Estimate token count by counting words (approximation)."""
        if not text:
            return 0
        return len(text.split())

    def transform_to_unified(server_response: dict, accumulated_content: str = "", prompt: Optional[str] = None) -> UnifiedCompletionResponse:
        response_type = server_response.get("type")
        content = server_response.get("content", "")
        finish_reason = None
        usage = None
        if response_type == "result":
            finish_reason = "length" if server_response.get(
                "truncated", False) else "stop"
            prompt_tokens = estimate_tokens(prompt)
            completion_tokens = estimate_tokens(accumulated_content)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        elif response_type == "error":
            logger.error(
                f"Server error: {server_response.get('message', 'Unknown error')}")
            raise HTTPException(status_code=500, detail=server_response.get(
                "message", "Server error"))
        return UnifiedCompletionResponse(
            id=server_response.get("prompt_id", str(uuid.uuid4())),
            created=int(time.time()),
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            prompt_id=server_response.get("prompt_id"),
            task_id=server_response.get("task_id")
        )
    if is_stream:
        def stream_chunks():
            accumulated_content = ""
            prompt = None
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    server_response = ParallelCompletionResponse(**chunk)
                    if prompt is None:
                        prompt = server_response.prompt
                    if server_response.content:
                        accumulated_content += server_response.content
                    unified_response = transform_to_unified(
                        server_response.dict(),
                        accumulated_content=accumulated_content,
                        prompt=prompt
                    )
                    logger.debug(
                        f"Streaming chunk: {unified_response.content}")
                    yield unified_response
                    if server_response.type == "result":
                        logger.info("Result chunk received, stopping stream")
                        return
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse chunk JSON: {e}, chunk: {line}")
                    raise HTTPException(
                        status_code=500, detail=f"Invalid JSON in streaming chunk: {str(e)}")
                except ValidationError as e:
                    logger.error(
                        f"Validation error for chunk: {e}, chunk: {line}")
                    raise HTTPException(
                        status_code=500, detail=f"Invalid response format: {str(e)}")
        return stream_chunks()
    response_text = response.text
    if not response_text.strip():
        logger.error("Empty response received from the server")
        raise HTTPException(
            status_code=500, detail="Empty response from MLX LM server")
    try:
        response_json = response.json()
        server_response = ParallelCompletionResponse(**response_json)
        unified_response = transform_to_unified(
            server_response.dict(),
            accumulated_content=server_response.content or "",
            prompt=server_response.prompt
        )
        logger.success(unified_response.content)
        return unified_response
    except json.JSONDecodeError as e:
        logger.error(
            f"JSON decode error: {e}, Response content: {response_text}")
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON response from server: {str(e)}")
    except ValidationError as e:
        logger.error(
            f"Validation error for response: {e}, Response content: {response_text}")
        raise HTTPException(
            status_code=500, detail=f"Invalid response format: {str(e)}")


def chat(request: ChatCompletionRequest) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    try:
        if isinstance(request.messages, str):
            request.messages = [Message(role="user", content=request.messages)]
        request_payload = request.dict(exclude_none=True)
        endpoint = "/chat" if request.stream else "/chat_non_stream"
        logger.info(f"Sending request to {BASE_URL}{endpoint}...")
        logger.gray("Request payload:")
        logger.debug(format_json(request_payload))
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )
        return _handle_response(response, is_stream=request.stream)
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(
            f"HTTP error: {str(e)}, Response content: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


def generate(request: TextCompletionRequest) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    try:
        if isinstance(request.prompt, str):
            request.prompt = [request.prompt]
        request_payload = request.dict(exclude_none=True)
        endpoint = "/generate" if request.stream else "/generate_non_stream"
        logger.info(f"Sending request to {BASE_URL}{endpoint}...")
        logger.gray("Request payload:")
        logger.debug(format_json(request_payload))
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )
        return _handle_response(response, is_stream=request.stream)
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(
            f"HTTP error: {str(e)}, Response content: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


def list_models() -> ModelsResponse:
    try:
        with MODEL_LIST_CACHE_LOCK:
            if MODEL_CACHE.get("models") is not None:  # Updated condition
                logger.info("Returning cached model list")
                return ModelsResponse(**MODEL_CACHE["models"])
        response = requests.get(
            f"{BASE_URL}/models",
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Response status code: {response.status_code}")
        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.error(f"Unexpected Content-Type: {content_type}")
            raise HTTPException(
                status_code=500, detail=f"Server returned non-JSON response: Content-Type {content_type}")
        response.raise_for_status()
        structured_response = ModelsResponse(**response.json())
        with MODEL_LIST_CACHE_LOCK:
            MODEL_CACHE["models"] = structured_response.dict()
        logger.success(format_json(structured_response.dict()))
        return structured_response
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(
            f"HTTP error: {str(e)}, Response content: {error_detail}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")
