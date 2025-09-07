from typing import Any, Literal, Union, List, Optional, Dict, Iterator
from pydantic.json_schema import JsonSchemaValue
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType, RoleMapping, Tool, ModelsResponse
from jet.llm.mlx.client import CompletionResponse, Message
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.base import MLX
from mlx_lm.server import PromptCache


def get_models() -> ModelsResponse:
    """Retrieve available models from the MLX model registry."""
    return MLXModelRegistry.get_models()


def chat(
    messages: Union[str, List[Message]],
    model: LLMModelType,
    draft_model: Optional[LLMModelType] = None,
    adapter: Optional[str] = None,
    max_tokens: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 0,
    top_k: int = 0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[RoleMapping] = None,
    tools: Optional[List[Tool]] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    client: Optional[MLX] = None,
    seed: Optional[int] = None,
    chat_template_args: Optional[ChatTemplateArgs] = None,
    prompt_cache: Optional[PromptCache] = None,
    response_format: Union[Literal["text", "json"], JsonSchemaValue] = "text",
) -> CompletionResponse:
    """Generate a chat completion."""
    if client is None:
        client = MLXModelRegistry.load_model(
            model=model,
            adapter_path=adapter,
            draft_model=draft_model,
            seed=seed,
        )
    return client.chat(
        messages=messages,
        model=model,
        draft_model=draft_model,
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        stop=stop,
        role_mapping=role_mapping,
        tools=tools,
        log_dir=log_dir,
        verbose=verbose,
        chat_template_args=chat_template_args,
        prompt_cache=prompt_cache,
        response_format=response_format
    )


def stream_chat(
    messages: Union[str, List[Message]],
    model: LLMModelType,
    draft_model: Optional[LLMModelType] = None,
    adapter: Optional[str] = None,
    max_tokens: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 0,
    top_k: int = 0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[RoleMapping] = None,
    tools: Optional[List[Tool]] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    client: Optional[MLX] = None,
    seed: Optional[int] = None,
    chat_template_args: Optional[ChatTemplateArgs] = None,
    prompt_cache: Optional[PromptCache] = None,
    response_format: Union[Literal["text", "json"], JsonSchemaValue] = "text",
) -> Iterator[CompletionResponse]:
    """Stream chat completions."""
    if client is None:
        client = MLXModelRegistry.load_model(
            model=model,
            adapter_path=adapter,
            draft_model=draft_model,
            seed=seed,
        )
    yield from client.stream_chat(
        messages=messages,
        model=model,
        draft_model=draft_model,
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        stop=stop,
        role_mapping=role_mapping,
        tools=tools,
        log_dir=log_dir,
        verbose=verbose,
        chat_template_args=chat_template_args,
        prompt_cache=prompt_cache,
        response_format=response_format
    )


def generate(
    prompt: str,
    model: LLMModelType,
    draft_model: Optional[LLMModelType] = None,
    adapter: Optional[str] = None,
    max_tokens: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 0,
    top_k: int = 0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    client: Optional[MLX] = None,
    seed: Optional[int] = None,
    prompt_cache: Optional[PromptCache] = None,
    response_format: Union[Literal["text", "json"], JsonSchemaValue] = "text",
) -> CompletionResponse:
    """Generate a text completion."""
    if client is None:
        client = MLXModelRegistry.load_model(
            model=model,
            adapter_path=adapter,
            draft_model=draft_model,
            seed=seed,
        )
    return client.generate(
        prompt=prompt,
        model=model,
        draft_model=draft_model,
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        stop=stop,
        log_dir=log_dir,
        verbose=verbose,
        prompt_cache=prompt_cache,
        response_format=response_format
    )


def stream_generate(
    prompt: str,
    model: LLMModelType,
    draft_model: Optional[LLMModelType] = None,
    adapter: Optional[str] = None,
    max_tokens: int = -1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 0,
    top_k: int = 0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]] = None,
    logprobs: int = -1,
    stop: Optional[Union[str, List[str]]] = None,
    log_dir: Optional[str] = None,
    verbose: bool = False,
    client: Optional[MLX] = None,
    seed: Optional[int] = None,
    prompt_cache: Optional[PromptCache] = None,
    response_format: Union[Literal["text", "json"], JsonSchemaValue] = "text",
) -> Iterator[CompletionResponse]:
    """Stream text completions."""
    if client is None:
        client = MLXModelRegistry.load_model(
            model=model,
            adapter_path=adapter,
            draft_model=draft_model,
            seed=seed,
        )
    yield from client.stream_generate(
        prompt=prompt,
        model=model,
        draft_model=draft_model,
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        xtc_probability=xtc_probability,
        xtc_threshold=xtc_threshold,
        logit_bias=logit_bias,
        logprobs=logprobs,
        stop=stop,
        log_dir=log_dir,
        verbose=verbose,
        prompt_cache=prompt_cache,
        response_format=response_format
    )


__all__ = [
    "chat",
    "stream_chat",
    "generate",
    "stream_generate",
]
