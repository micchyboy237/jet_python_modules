from typing import AsyncIterator
from typing import Callable, Dict, List, Optional, Union, Literal, Iterator, Any
from jet.llm.mlx.remote.client import MLXRemoteClient
from jet.llm.mlx.remote.types import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    ModelsResponse,
    HealthResponse,
)
from jet.llm.mlx.remote.utils import (
    prepare_chat_request,
    prepare_text_request,
    process_chat_response,
    process_text_response,
    process_stream_chat_response,
    process_stream_text_response,
    save_logs,
)
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.mlx_utils import has_tools
from jet.logger import logger
from pydantic.json_schema import JsonSchemaValue


def chat(
    messages: Union[str, List[Message]],
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[List[Callable]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> ChatCompletionResponse:
    """Create a chat completion via the remote server with tool usage and response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)
    history = history or ChatHistory()

    req = prepare_chat_request(
        messages=messages,
        history=history,
        system_prompt=system_prompt,
        with_history=with_history,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        role_mapping=role_mapping,
        tools=tools,
        stream=False
    )

    response = list(client.create_chat_completion(req, stream=False))[0]
    processed_response = process_chat_response(
        response, history, with_history, tools)
    save_logs(
        messages,
        processed_response,
        model,
        tools,
        "chat",
        history=history,
        system_prompt=system_prompt,
        with_history=with_history,
        response_format=response_format,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        role_mapping=role_mapping,
        stream=False
    )
    return processed_response


def stream_chat(
    messages: Union[str, List[Message]],
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[List[Callable]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> Iterator[ChatCompletionResponse]:
    """Stream chat completion chunks via the remote server with tool usage and response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)
    history = history or ChatHistory()

    req = prepare_chat_request(
        messages=messages,
        history=history,
        system_prompt=system_prompt,
        with_history=with_history,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        role_mapping=role_mapping,
        tools=tools,
        stream=True
    )

    chunks = client.create_chat_completion(req, stream=True)
    last_response = None
    for chunk in process_stream_chat_response(chunks, history, with_history, tools):
        last_response = chunk
        yield chunk

    if last_response:
        save_logs(
            messages,
            last_response,
            model,
            tools,
            "stream_chat",
            history=history,
            system_prompt=system_prompt,
            with_history=with_history,
            response_format=response_format,
            draft_model=draft_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            logit_bias=logit_bias,
            logprobs=logprobs,
            seed=seed,
            stop=stop,
            role_mapping=role_mapping,
            stream=True
        )


async def process_stream_chat_response(
    chunks: Iterator[ChatCompletionResponse],
    history: ChatHistory,
    with_history: bool,
    tools: Optional[List[Callable]] = None,
) -> AsyncIterator[ChatCompletionResponse]:
    """Process stream chat response chunks asynchronously."""
    for chunk in chunks:
        # Process chunk if needed (e.g., update history or handle tools)
        if with_history and history:
            history.add_response(chunk)
        yield chunk


async def astream_chat(
    messages: Union[str, List[Message]],
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[List[Callable]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> AsyncIterator[ChatCompletionResponse]:
    """Asynchronously stream chat completion chunks via the remote server with tool usage and response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)
    history = history or ChatHistory()
    req = prepare_chat_request(
        messages=messages,
        history=history,
        system_prompt=system_prompt,
        with_history=with_history,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        role_mapping=role_mapping,
        tools=tools,
        stream=True
    )
    chunks = client.create_chat_completion(req, stream=True)
    last_response = None
    async for chunk in process_stream_chat_response(chunks, history, with_history, tools):
        last_response = chunk
        yield chunk
    if last_response:
        save_logs(
            messages,
            last_response,
            model,
            tools,
            "astream_chat",
            history=history,
            system_prompt=system_prompt,
            with_history=with_history,
            response_format=response_format,
            draft_model=draft_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            logit_bias=logit_bias,
            logprobs=logprobs,
            seed=seed,
            stop=stop,
            role_mapping=role_mapping,
            stream=True
        )


def generate(
    prompt: str,
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> TextCompletionResponse:
    """Create a text completion via the remote server with response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)

    req = prepare_text_request(
        prompt=prompt,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        stream=False
    )

    response = list(client.create_text_completion(req, stream=False))[0]
    processed_response = process_text_response(response)
    save_logs(
        prompt,
        processed_response,
        model,
        None,  # tools
        "generate",
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        response_format=response_format,
        stream=False
    )
    return processed_response


def stream_generate(
    prompt: str,
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> Iterator[TextCompletionResponse]:
    """Stream text completion chunks via the remote server with response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)

    req = prepare_text_request(
        prompt=prompt,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        stream=True
    )

    chunks = client.create_text_completion(req, stream=True)
    last_response = None
    for chunk in process_stream_text_response(chunks):
        last_response = chunk
        yield chunk

    if last_response:
        save_logs(
            prompt,
            last_response,
            model,
            None,  # tools
            "stream_generate",
            draft_model=draft_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            logit_bias=logit_bias,
            logprobs=logprobs,
            seed=seed,
            stop=stop,
            response_format=response_format,
            stream=True
        )


async def process_stream_text_response(
    chunks: Iterator[TextCompletionResponse]
) -> AsyncIterator[TextCompletionResponse]:
    """Process stream text response chunks asynchronously."""
    for chunk in chunks:
        yield chunk


async def astream_generate(
    prompt: str,
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
    response_format: Optional[Union[Literal["text",
                                            "json"], JsonSchemaValue]] = None,
) -> AsyncIterator[TextCompletionResponse]:
    """Asynchronously stream text completion chunks via the remote server with response format support."""
    client = client or MLXRemoteClient(base_url=base_url, verbose=verbose)
    req = prepare_text_request(
        prompt=prompt,
        response_format=response_format,
        model=model,
        draft_model=draft_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        seed=seed,
        stop=stop,
        stream=True
    )
    chunks = client.create_text_completion(req, stream=True)
    last_response = None
    async for chunk in process_stream_text_response(chunks):
        last_response = chunk
        yield chunk
    if last_response:
        save_logs(
            prompt,
            last_response,
            model,
            None,
            "astream_generate",
            draft_model=draft_model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            logit_bias=logit_bias,
            logprobs=logprobs,
            seed=seed,
            stop=stop,
            response_format=response_format,
            stream=True
        )


__all__ = [
    "chat",
    "stream_chat",
    "astream_chat",
    "generate",
    "stream_generate",
    "astream_generate",
]
