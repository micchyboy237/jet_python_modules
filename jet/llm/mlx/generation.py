import json
import time
import uuid
from jet.llm.mlx.mlx_types import ModelKey
from jet.transformers.formatters import format_json
import requests
from typing import List, Dict, Optional, Union, Literal, Generator
from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException
from jet.logger import logger
BASE_URL = "http://localhost:9000"


class Message(BaseModel):
    role: str = Field(...,
                      description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")


class Delta(BaseModel):
    role: Optional[str] = Field(
        None, description="Role of the message sender in streaming delta")
    content: Optional[str] = Field(
        None, description="Content of the message in streaming delta")


class BaseCompletionRequest(BaseModel):
    temperature: Optional[float] = Field(
        default=1.0, description="Sampling temperature", ge=0.0)
    top_p: Optional[float] = Field(
        default=1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(
        default=100, description="Maximum number of tokens to generate", ge=1)
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Sequences where generation should stop")
    repetition_penalty: Optional[float] = Field(
        default=1.0, description="Penalty for repeated tokens", ge=1.0)
    repetition_context_size: Optional[int] = Field(
        default=20, description="Context window size for repetition penalty", ge=1)
    logit_bias: Optional[Dict[int, float]] = Field(
        default=None, description="Token ID to bias value mapping")
    logprobs: Optional[int] = Field(
        default=None, description="Number of top tokens and log probabilities to return", ge=1, le=10)
    model: Optional[str] = Field(
        default=None, description="Path to local model or Hugging Face repo ID")
    adapters: Optional[str] = Field(
        default=None, description="Path to low-rank adapters")
    draft_model: Optional[str] = Field(
        default=None, description="Smaller model for speculative decoding")
    num_draft_tokens: Optional[int] = Field(
        default=3, description="Number of draft tokens for draft model", ge=1)


class ChatCompletionRequest(BaseCompletionRequest):
    messages: List[Message] = Field(
        ..., description="Array of message objects representing conversation history")
    role_mapping: Optional[Dict[str, str]] = Field(
        default=None, description="Custom role prefixes for prompt generation")


class TextCompletionRequest(BaseCompletionRequest):
    prompt: str = Field(..., description="Input prompt for text completion")


class Usage(BaseModel):
    prompt_tokens: int = Field(...,
                               description="Number of prompt tokens processed")
    completion_tokens: int = Field(...,
                                   description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total number of tokens")


class UnifiedCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    created: int = Field(...,
                         description="Timestamp for when the request was processed")
    content: Optional[str] = Field(
        None, description="Generated content (text or message)")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Reason the generation ended")
    usage: Optional[Usage] = Field(None, description="Token usage information")
    prompt_id: Optional[str] = Field(
        None, description="Unique identifier for the prompt")
    task_id: Optional[str] = Field(
        None, description="Unique identifier for the task")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Hugging Face repo ID")
    short_name: str = Field(..., description="Model key")
    object: Optional[str] = Field(
        "model", description="Type of object, default is 'model'")
    created: int = Field(..., description="Timestamp for model creation")


class ModelsResponse(BaseModel):
    object: str = Field("list", description="Type of response, always 'list'")
    data: List[ModelInfo] = Field(..., description="List of available models")


class ParallelCompletionResponse(BaseModel):
    type: Literal["chunk", "result",
                  "error"] = Field(..., description="Type of response")
    prompt: Optional[str] = Field(
        None, description="Original prompt or message")
    content: Optional[str] = Field(None, description="Generated content")
    prompt_id: Optional[str] = Field(
        None, description="Unique identifier for the prompt")
    task_id: Optional[str] = Field(
        None, description="Unique identifier for the task")
    truncated: Optional[bool] = Field(
        None, description="Whether the response was truncated")
    message: Optional[str] = Field(
        None, description="Error message, if type is error")


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
            # Compute usage for final response
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
                        prompt = server_response.prompt  # Set prompt from first chunk
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
        logger.error(
            f"HTTP error: {str(e)}, Response content: {e.response.text if e.response else 'N/A'}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {str(e)}")
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
        request_payload = request.dict(exclude_none=True)
        # Map model repo ID to short_name from /models endpoint
        if request_payload.get("model"):
            models_response = list_models()
            for model_info in models_response.data:
                if model_info.id == request_payload["model"]:
                    request_payload["model"] = model_info.short_name
                    break
            else:
                logger.error(
                    f"Model {request_payload['model']} not found in available models")
                raise HTTPException(
                    status_code=400, detail=f"Model {request_payload['model']} not found")
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
        logger.error(
            f"HTTP error: {str(e)}, Response content: {e.response.text if e.response else 'N/A'}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {str(e)}")
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
        logger.success(format_json(structured_response.dict()))
        return structured_response
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {e.response.text if e.response else 'N/A'}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")
