import json
from jet.transformers.formatters import format_json
import requests
from typing import List, Dict, Optional, Union, Literal, Generator
from pydantic import BaseModel, Field
from fastapi import HTTPException
from jet.logger import logger

BASE_URL = "http://localhost:8003/v1"

# Request Models


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

# Response Models


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
        None, description="Reason the completion ended")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Hugging Face repo ID")
    created: int = Field(..., description="Timestamp for model creation")


class ModelsResponse(BaseModel):
    data: List[ModelInfo] = Field(..., description="List of available models")

# Internal Response Models (for parsing server responses)


class LogProbs(BaseModel):
    token_logprobs: List[float] = Field(
        default_factory=list, description="Log probabilities for generated tokens")
    tokens: Optional[List[int]] = Field(
        default=None, description="Generated token IDs")
    top_logprobs: List[Dict[int, float]] = Field(
        default_factory=list, description="Top tokens and their log probabilities")


class Choice(BaseModel):
    index: int = Field(..., description="Index of the choice in the list")
    message: Optional[Message] = Field(
        None, description="Text response from the model for non-streaming chat completions")
    text: Optional[str] = Field(
        None, description="Generated text for text completion")
    delta: Optional[Delta] = Field(
        None, description="Delta response for streaming chat completions")
    logprobs: Optional[LogProbs] = Field(
        None, description="Log probabilities for generated tokens")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Reason the completion ended")


class ServerCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the chat")
    system_fingerprint: str = Field(...,
                                    description="Unique identifier for the system")
    object: Literal["chat.completion", "chat.completion.chunk", "text.completion",
                    "text.completion.chunk", "text_completion"] = Field(..., description="Type of response")
    created: int = Field(...,
                         description="Timestamp for when the request was processed")
    choices: List[Choice] = Field(..., description="List of output choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")

# Helper Function


def _handle_response(response: requests.Response, is_stream: bool) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response headers: {response.headers}")

    content_type = response.headers.get("Content-Type", "")
    expected_content_type = "text/event-stream" if is_stream else "application/json"
    if expected_content_type not in content_type:
        logger.error(
            f"Unexpected Content-Type: {content_type}, expected {expected_content_type}")
        raise HTTPException(
            status_code=500, detail=f"Server returned unexpected Content-Type: {content_type}")

    response.raise_for_status()

    def transform_to_unified(server_response: dict) -> UnifiedCompletionResponse:
        choices = server_response.get("choices", [])
        content = ""
        finish_reason = None
        if choices:
            choice = choices[0]
            if choice.get("delta") and choice["delta"].get("content"):
                content = choice["delta"]["content"]
            elif choice.get("message") and choice["message"].get("content"):
                content = choice["message"]["content"]
            elif choice.get("text"):
                content = choice["text"]
            finish_reason = choice.get("finish_reason")

        return UnifiedCompletionResponse(
            id=server_response["id"],
            created=server_response["created"],
            content=content,
            finish_reason=finish_reason,
            usage=Usage(
                **server_response["usage"]) if server_response.get("usage") else None
        )

    if is_stream:
        def stream_chunks():
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                if not line.startswith("data: "):
                    logger.error(f"Invalid SSE chunk format: {line}")
                    raise HTTPException(
                        status_code=500, detail="Invalid server-sent event format")
                json_data = line[len("data: "):].strip()
                if not json_data:
                    logger.error("Empty JSON data in SSE chunk")
                    continue
                try:
                    chunk = json.loads(json_data)
                    for choice in chunk.get("choices", []):
                        if choice.get("logprobs") and choice["logprobs"].get("tokens") is None:
                            choice["logprobs"]["tokens"] = []
                    server_response = ServerCompletionResponse(**chunk)
                    unified_response = transform_to_unified(
                        server_response.dict())
                    logger.success(unified_response.content, flush=True)
                    yield unified_response
                    if any(choice.get("finish_reason") for choice in chunk.get("choices", [])):
                        logger.newline()
                        logger.info(
                            "Finish reason detected in chunk, stopping stream")
                        return
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse chunk JSON: {e}, chunk: {json_data}")
                    raise HTTPException(
                        status_code=500, detail=f"Invalid JSON in streaming chunk: {str(e)}")
        return stream_chunks()

    response_text = response.text
    if not response_text.strip():
        logger.error("Empty response received from the server")
        raise HTTPException(
            status_code=500, detail="Empty response from MLX LM server")
    try:
        response_json = response.json()
        for choice in response_json.get("choices", []):
            if choice.get("logprobs") and choice["logprobs"].get("tokens") is None:
                choice["logprobs"]["tokens"] = []
        server_response = ServerCompletionResponse(**response_json)
        unified_response = transform_to_unified(server_response.dict())
        logger.success(unified_response.content)
        return unified_response
    except requests.exceptions.JSONDecodeError as e:
        logger.error(
            f"JSON decode error: {str(e)}, Response content: {response_text}")
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON response from server: {str(e)}")

# API Calls


def chat_completions(request: ChatCompletionRequest) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    try:
        request_payload = request.dict(exclude_none=True)
        logger.info(f"Sending request to {BASE_URL}/chat/completions...")
        logger.gray("Request payload:")
        logger.debug(format_json(request_payload))
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )
        return _handle_response(response, is_stream=request.stream)
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
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


def text_completions(request: TextCompletionRequest) -> Union[UnifiedCompletionResponse, Generator[UnifiedCompletionResponse, None, None]]:
    try:
        request_payload = request.dict(exclude_none=True)
        logger.info(f"Sending request to {BASE_URL}/completions...")
        logger.gray("Request payload:")
        logger.debug(format_json(request_payload))
        response = requests.post(
            f"{BASE_URL}/completions",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )
        return _handle_response(response, is_stream=request.stream)
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
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
        logger.success(format_json(structured_response.model_dump()))
        return structured_response
    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
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
