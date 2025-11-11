import argparse
import uuid
import json
import time

from pathlib import Path
from pydantic.json_schema import JsonSchemaValue
from jet.llm.mlx.helpers.detect_repetition import NgramRepeat
from jet.llm.mlx.mlx_types import ChatTemplateArgs
from jet.llm.mlx.mlx_utils import process_response_format
from typing import Dict, List, Optional, Tuple, Union, Literal, Any, Iterator
from dataclasses import dataclass, field
from huggingface_hub import scan_cache_dir
from jet.llm.mlx.config import DEFAULT_LOG_DIR, DEFAULT_MODEL
from jet.logger import logger
from jet.transformers.formatters import format_json
import mlx.core as mx
from mlx_lm.server import ModelProvider, get_system_fingerprint, convert_chat, process_message_content
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from jet.llm.mlx.utils.logit_bias import convert_logit_bias
from jet.llm.logger_utils import ChatLogger
from jet.models.model_types import (
    MLXTokenizer,
    Message,
    Tool,
    RoleMapping,
    CompletionResponse,
    ModelsResponse,
    ModelInfo,
    LLMModelType,
    LLMModelKey,
    LLMModelValue,
)
from jet.models.utils import resolve_model_value

DEFAULT_CHAT_TEMPLATE_ARGS: ChatTemplateArgs = {
    "add_generation_prompt": True,
    # Switches between thinking and non-thinking modes. Default is False.
    "enable_thinking": False,
}


@dataclass
class Config:
    model: Optional[LLMModelType] = None
    adapter_path: Optional[str] = None
    draft_model: Optional[LLMModelType] = None
    trust_remote_code: bool = False
    chat_template: Optional[str] = None
    use_default_chat_template: bool = True
    chat_template_args: Optional[ChatTemplateArgs] = field(
        default_factory=lambda: DEFAULT_CHAT_TEMPLATE_ARGS.copy()
    )
    max_tokens: int = -1
    temperature: float = 0.0
    top_p: float = 1.0
    min_p: float = 0.0
    min_tokens_to_keep: int = 0
    top_k: int = 0
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    logit_bias: Optional[Union[Dict[int, float],
                               Dict[str, float], str, List[str]]] = None
    logprobs: int = -1
    stop: Optional[Union[str, List[str]]] = None
    response_format: Union[Literal["text", "json"], JsonSchemaValue] = "text"
    verbose: bool = False


class MLXLMClient:
    """A client for interacting with MLX-LM models directly in Python."""

    @staticmethod
    def get_models() -> ModelsResponse:
        """List available local models."""
        files: List[str] = ["config.json",
                            "model.safetensors.index.json", "tokenizer_config.json"]

        def probably_mlx_lm(repo: Any) -> bool:
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded MLX models
        hf_cache_info = scan_cache_dir()
        downloaded_models: List[Any] = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models with creation time from repo_path
        models: List[ModelInfo] = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": int(repo.repo_path.stat().st_ctime) if isinstance(repo.repo_path, Path) else None,
                "modified": int(repo.last_modified),
            }
            for repo in downloaded_models
        ]

        return {"object": "list", "data": models}

    def __init__(
        self,
        model: Optional[LLMModelType] = None,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        chat_template_args: Optional[ChatTemplateArgs] = None,
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
        response_format: Union[Literal["text", "json"],
                               JsonSchemaValue] = "text",
        verbose: bool = False,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        device: Optional[Literal["cpu", "mps"]] = "mps",
        prompt_cache: Optional[List[Any]] = None
    ) -> None:
        """Initialize the client with configuration, generation parameters, response format, and verbosity."""
        if device and device not in ["cpu", "mps"]:
            logger.error(f"Unsupported device {device} for MLX model {model}")
            raise ValueError(
                f"Device {device} is not supported for MLX (use 'cpu' or 'mps')")
        if device == "cpu":
            mx.set_default_device(mx.cpu)
        else:
            mx.set_default_device(mx.gpu)
        if seed:
            mx.random.seed(seed)
        model_value = resolve_model_value(model) if model else None
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None
        merged_chat_template_args = DEFAULT_CHAT_TEMPLATE_ARGS.copy()
        if chat_template_args is not None:
            merged_chat_template_args.update(chat_template_args)
        self._validate_parameters(
            max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model_value, adapter_path, response_format
        )
        config = Config(
            model=model_value,
            adapter_path=adapter_path,
            draft_model=draft_model_value,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            chat_template_args=merged_chat_template_args,
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
            response_format=response_format,
            verbose=verbose
        )
        self.cli_args: argparse.Namespace = argparse.Namespace(
            model=config.model,
            adapter_path=config.adapter_path,
            draft_model=config.draft_model,
            trust_remote_code=config.trust_remote_code,
            chat_template=config.chat_template,
            use_default_chat_template=config.use_default_chat_template,
            chat_template_args=config.chat_template_args,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            min_tokens_to_keep=config.min_tokens_to_keep,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            repetition_context_size=config.repetition_context_size,
            xtc_probability=config.xtc_probability,
            xtc_threshold=config.xtc_threshold,
            logit_bias=config.logit_bias,
            logprobs=config.logprobs,
            stop=config.stop,
            response_format=config.response_format,
            verbose=config.verbose
        )
        self.model_provider: ModelProvider = ModelProvider(self.cli_args)
        self.prompt_cache: List[Any] = prompt_cache if prompt_cache is not None else [
        ]
        self.system_fingerprint: str = get_system_fingerprint()
        self.created: int = int(time.time())
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.model = self.model_provider.model
        self.tokenizer: MLXTokenizer = self.model_provider.tokenizer
        self._chat_template_args = config.chat_template_args

    def __call__(self, *args, **kwargs) -> Union[mx.array, Tuple[mx.array, Any]]:
        """
        Call the underlying MLX model to generate logits.

        Args:
            *args: Positional arguments to pass to the model (e.g., input token arrays).
            **kwargs: Keyword arguments to pass to the model (e.g., attention masks).

        Returns:
            Union[mx.array, Tuple[mx.array, Any]]: The logits or a tuple of logits and additional outputs
            (e.g., hidden states) depending on the model's configuration.

        Raises:
            ValueError: If the model is not loaded or inputs are invalid.
        """
        if self.model is None:
            logger.error("Model is not loaded")
            raise ValueError("Model is not loaded")

        logger.debug("Calling model with args: %s, kwargs: %s", args, kwargs)
        return self.model(*args, **kwargs)

    def __del__(self) -> None:
        """Clean up resources when the instance is destroyed."""
        logger.debug("Cleaning up MLXLMClient instance")
        try:
            # self.reset_model()
            pass
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            # Ensure MLX cache is cleared
            mx.clear_cache()

    def chat(
        self,
        messages: List[Message],
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        min_tokens_to_keep: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None,
        log_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
        chat_template_args: Optional[ChatTemplateArgs] = None,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Optional[Union[Literal["text",
                                                "json"], JsonSchemaValue]] = None
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion."""
        model_value = resolve_model_value(model)
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None
        active_response_format = response_format if response_format is not None else self.cli_args.response_format
        active_verbose = verbose if verbose is not None else self.cli_args.verbose
        self._validate_parameters(
            max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature if temperature is not None else self.cli_args.temperature,
            top_p if top_p is not None else self.cli_args.top_p,
            repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs if logprobs is not None else self.cli_args.logprobs,
            model_value, adapter, active_response_format
        )
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)
        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")
        stop_words: List[str] = (
            [stop] if isinstance(
                stop, str) else stop or self.cli_args.stop or []
        )
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]
        request_id: str = f"chatcmpl-{uuid.uuid4()}"
        object_type: str = "chat.completion"
        modified_messages = process_response_format(messages, active_response_format)
        if role_mapping:
            prompt_str: str = convert_chat(modified_messages, role_mapping)
            prompt = tokenizer.encode(prompt_str)
        elif tokenizer.chat_template:
            process_message_content(modified_messages)
            chat_template_settings: ChatTemplateArgs = {
                **(self._chat_template_args or {}),
                **(chat_template_args or {})
            }
            if active_verbose:
                logger.newline()
                logger.info("Chat Template Args:")
                logger.debug(format_json(chat_template_settings))
            prompt: List[int] = tokenizer.apply_chat_template(
                modified_messages,
                tools,
                **chat_template_settings
            )
        active_prompt_cache = prompt_cache if prompt_cache is not None else self.prompt_cache
        response = self._generate_completion(
            prompt=prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature=temperature if temperature is not None else self.cli_args.temperature,
            top_p=top_p if top_p is not None else self.cli_args.top_p,
            min_p=min_p if min_p is not None else self.cli_args.min_p,
            min_tokens_to_keep=min_tokens_to_keep if min_tokens_to_keep is not None else self.cli_args.min_tokens_to_keep,
            top_k=top_k if top_k is not None else self.cli_args.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias=logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=active_verbose,
            prompt_cache=active_prompt_cache,
            response_format=active_response_format
        )
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="chat").log_interaction(
                modified_messages,
                response,
                model=model,
                tools=tools,
                max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
                temperature=temperature if temperature is not None else self.cli_args.temperature,
                top_p=top_p if top_p is not None else self.cli_args.top_p,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
                repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
                xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
                xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
                logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            )
        return response

    def stream_chat(
        self,
        messages: List[Message],
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        min_tokens_to_keep: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None,
        log_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
        chat_template_args: Optional[ChatTemplateArgs] = None,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Optional[Union[Literal["text",
                                                "json"], JsonSchemaValue]] = None
    ) -> Iterator[CompletionResponse]:
        """Stream chat completions as they are generated."""
        model_value = resolve_model_value(model)
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None
        active_response_format = response_format if response_format is not None else self.cli_args.response_format
        active_verbose = verbose if verbose is not None else self.cli_args.verbose
        self._validate_parameters(
            max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature if temperature is not None else self.cli_args.temperature,
            top_p if top_p is not None else self.cli_args.top_p,
            repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs if logprobs is not None else self.cli_args.logprobs,
            model_value, adapter, active_response_format
        )
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)
        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")
        stop_words: List[str] = (
            [stop] if isinstance(
                stop, str) else stop or self.cli_args.stop or []
        )
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]
        request_id: str = f"chatcmpl-{uuid.uuid4()}"
        object_type: str = "chat.completion"
        modified_messages = process_response_format(messages, active_response_format)
        if role_mapping:
            prompt_str: str = convert_chat(modified_messages, role_mapping)
            prompt = tokenizer.encode(prompt_str)
        elif tokenizer.chat_template:
            process_message_content(modified_messages)
            chat_template_settings: ChatTemplateArgs = {
                **(self._chat_template_args or {}),
                **(chat_template_args or {})
            }
            if active_verbose:
                logger.newline()
                logger.info("Chat Template Args:")
                logger.debug(format_json(chat_template_settings))
            prompt: List[int] = tokenizer.apply_chat_template(
                modified_messages,
                tools,
                **chat_template_settings
            )
        active_prompt_cache = prompt_cache if prompt_cache is not None else self.prompt_cache
        for response in self._stream_generate_completion(
            prompt=prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature=temperature if temperature is not None else self.cli_args.temperature,
            top_p=top_p if top_p is not None else self.cli_args.top_p,
            min_p=min_p if min_p is not None else self.cli_args.min_p,
            min_tokens_to_keep=min_tokens_to_keep if min_tokens_to_keep is not None else self.cli_args.min_tokens_to_keep,
            top_k=top_k if top_k is not None else self.cli_args.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias=logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=active_verbose,
            prompt_cache=active_prompt_cache,
            response_format=active_response_format
        ):
            yield response
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_chat").log_interaction(
                modified_messages,
                response,
                model=model,
                tools=tools,
                max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
                temperature=temperature if temperature is not None else self.cli_args.temperature,
                top_p=top_p if top_p is not None else self.cli_args.top_p,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
                repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
                xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
                xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
                logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            )

    def generate(
        self,
        prompt: str,
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        min_tokens_to_keep: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        log_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Optional[Union[Literal["text",
                                                "json"], JsonSchemaValue]] = None
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a text completion."""
        model_value = resolve_model_value(model)
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None
        active_response_format = response_format if response_format is not None else self.cli_args.response_format
        active_verbose = verbose if verbose is not None else self.cli_args.verbose
        self._validate_parameters(
            max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature if temperature is not None else self.cli_args.temperature,
            top_p if top_p is not None else self.cli_args.top_p,
            repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs if logprobs is not None else self.cli_args.logprobs,
            model_value, adapter, active_response_format
        )
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)
        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")
        stop_words: List[str] = (
            [stop] if isinstance(
                stop, str) else stop or self.cli_args.stop or []
        )
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]
        request_id: str = f"cmpl-{uuid.uuid4()}"
        object_type: str = "text.completion"
        modified_prompt = process_response_format(prompt, active_response_format)
        prompt_tokens: List[int] = tokenizer.encode(modified_prompt)
        response = self._generate_completion(
            prompt=prompt_tokens,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature=temperature if temperature is not None else self.cli_args.temperature,
            top_p=top_p if top_p is not None else self.cli_args.top_p,
            min_p=min_p if min_p is not None else self.cli_args.min_p,
            min_tokens_to_keep=min_tokens_to_keep if min_tokens_to_keep is not None else self.cli_args.min_tokens_to_keep,
            top_k=top_k if top_k is not None else self.cli_args.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias=logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=active_verbose,
            prompt_cache=prompt_cache,
            response_format=active_response_format
        )
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="generate").log_interaction(
                modified_prompt,
                response,
                model=model,
                max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
                temperature=temperature if temperature is not None else self.cli_args.temperature,
                top_p=top_p if top_p is not None else self.cli_args.top_p,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
                repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
                xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
                xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
                logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            )
        return response

    def stream_generate(
        self,
        prompt: str,
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        min_tokens_to_keep: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = None,
        xtc_probability: Optional[float] = None,
        xtc_threshold: Optional[float] = None,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        log_dir: Optional[str] = None,
        verbose: Optional[bool] = None,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Optional[Union[Literal["text",
                                                "json"], JsonSchemaValue]] = None
    ) -> Iterator[CompletionResponse]:
        """Stream text completions as they are generated."""
        model_value = resolve_model_value(model)
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None
        active_response_format = response_format if response_format is not None else self.cli_args.response_format
        active_verbose = verbose if verbose is not None else self.cli_args.verbose
        self._validate_parameters(
            max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature if temperature is not None else self.cli_args.temperature,
            top_p if top_p is not None else self.cli_args.top_p,
            repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs if logprobs is not None else self.cli_args.logprobs,
            model_value, adapter, active_response_format
        )
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)
        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")
        stop_words: List[str] = (
            [stop] if isinstance(
                stop, str) else stop or self.cli_args.stop or []
        )
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]
        request_id: str = f"cmpl-{uuid.uuid4()}"
        object_type: str = "text.completion"
        modified_prompt = process_response_format(prompt, active_response_format)
        prompt_tokens: List[int] = tokenizer.encode(modified_prompt)
        for response in self._stream_generate_completion(
            prompt=prompt_tokens,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
            temperature=temperature if temperature is not None else self.cli_args.temperature,
            top_p=top_p if top_p is not None else self.cli_args.top_p,
            min_p=min_p if min_p is not None else self.cli_args.min_p,
            min_tokens_to_keep=min_tokens_to_keep if min_tokens_to_keep is not None else self.cli_args.min_tokens_to_keep,
            top_k=top_k if top_k is not None else self.cli_args.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
            repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
            xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
            xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
            logit_bias=logit_bias if logit_bias is not None else self.cli_args.logit_bias,
            logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=active_verbose,
            prompt_cache=prompt_cache,
            response_format=active_response_format
        ):
            yield response
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_generate").log_interaction(
                modified_prompt,
                response,
                model=model,
                max_tokens=max_tokens if max_tokens is not None else self.cli_args.max_tokens,
                temperature=temperature if temperature is not None else self.cli_args.temperature,
                top_p=top_p if top_p is not None else self.cli_args.top_p,
                repetition_penalty=repetition_penalty if repetition_penalty is not None else self.cli_args.repetition_penalty,
                repetition_context_size=repetition_context_size if repetition_context_size is not None else self.cli_args.repetition_context_size,
                xtc_probability=xtc_probability if xtc_probability is not None else self.cli_args.xtc_probability,
                xtc_threshold=xtc_threshold if xtc_threshold is not None else self.cli_args.xtc_threshold,
                logprobs=logprobs if logprobs is not None else self.cli_args.logprobs,
            )

    def _validate_parameters(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: Optional[float],
        repetition_context_size: int,
        xtc_probability: float,
        xtc_threshold: float,
        logit_bias: Optional[Dict[int, float]],
        logprobs: int,
        model: LLMModelType,
        adapter: Optional[str],
        response_format: Union[Literal["text", "json"], JsonSchemaValue]
    ) -> None:
        """Validate model parameters."""
        if not isinstance(max_tokens, int) or max_tokens < -1:
            raise ValueError(
                "max_tokens must be '-1' (context length) or a positive integer")
        if not isinstance(temperature, (float, int)) or temperature < 0:
            raise ValueError("temperature must be a non-negative float")
        if not isinstance(top_p, (float, int)) or top_p < 0 or top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")
        if repetition_penalty is not None:
            if not isinstance(repetition_penalty, (float, int)) or repetition_penalty < 0:
                raise ValueError(
                    "repetition_penalty must be a non-negative float")
        if logprobs != -1 and not (0 < logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {logprobs}")
        if not isinstance(repetition_context_size, int) or repetition_context_size < 0:
            raise ValueError(
                "repetition_context_size must be a non-negative integer")
        if not isinstance(xtc_probability, float) or not 0.0 <= xtc_probability <= 1.0:
            raise ValueError(
                "xtc_probability must be a float between 0.0 and 1.0")
        if not isinstance(xtc_threshold, float) or not 0.0 <= xtc_threshold <= 0.5:
            raise ValueError(
                "xtc_threshold must be a float between 0.0 and 0.5")
        if not isinstance(model, (str, LLMModelKey, LLMModelValue)):
            raise ValueError("model must be a string or valid model type")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError("adapter must be a string")
        if logit_bias is not None:
            if not isinstance(logit_bias, (dict, str, list)):
                raise ValueError(
                    "logit_bias must be a dict of int to float, str or list[str]")
        if not isinstance(response_format, (str, dict)):
            raise ValueError(
                "response_format must be 'text', 'json', or a JsonSchemaValue dict")
        if isinstance(response_format, str) and response_format not in ["text", "json"]:
            raise ValueError("response_format string must be 'text' or 'json'")

    def _generate_response(
        self,
        text: str,
        finish_reason: Optional[Literal["length", "stop"]],
        request_id: str,
        object_type: str,
        model: str,
        segment: Optional[str] = None,
        prompt_token_count: Optional[int] = None,
        prompt_tps: Optional[float] = None,
        completion_token_count: Optional[int] = None,
        completion_tps: Optional[float] = None,
        peak_memory: Optional[float] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
        repetitions: Optional[List[NgramRepeat]] = None,
        response_format: Union[Literal["text", "json"],
                               JsonSchemaValue] = "text"
    ) -> CompletionResponse:
        """Generate a response packet in OpenAI-compatible format."""
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tokens = tokens or []
        active_response_format = response_format
        # Handle JSON or JsonSchemaValue response format
        if isinstance(response_format, (str, dict)) and (response_format == "json" or isinstance(response_format, dict)):
            try:
                # Attempt to parse and re-encode to ensure valid JSON
                parsed = json.loads(text) if text else {}
                if isinstance(response_format, dict):
                    # Basic validation against JsonSchemaValue; assumes model output matches schema
                    # In a real implementation, use pydantic for schema validation
                    pass
                text = json.dumps(parsed, ensure_ascii=False)
                segment = json.dumps(json.loads(segment),
                                     ensure_ascii=False) if segment else None
            except json.JSONDecodeError:
                active_response_format = "text"
        response: CompletionResponse = {
            "id": request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": object_type,
            "model": model,
            "created": self.created,
            "usage": None,
            "content": text,
            "repetitions": repetitions,
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "tokens": tokens,
                    },
                    "finish_reason": finish_reason,
                    "message": None,
                    "delta": None,
                    "text": None
                }
            ],
        }
        if not (isinstance(prompt_token_count, int) and isinstance(completion_token_count, int)):
            raise ValueError(
                "Response type is complete, but token counts not provided")
        response["usage"] = {
            "prompt_tokens": prompt_token_count,
            "prompt_tps": prompt_tps,
            "completion_tokens": completion_token_count,
            "completion_tps": completion_tps,
            "peak_memory": peak_memory,
            "total_tokens": prompt_token_count + completion_token_count,
        }
        choice = response["choices"][0]
        if "chat.completion" in object_type:
            choice["message"] = {
                "role": "assistant", "content": segment if segment is not None else text}
        elif "text.completion" in object_type:
            choice["text"] = segment if segment is not None else text
        else:
            raise ValueError(f"Unsupported response type: {object_type}")
        return response

    def _get_prompt_cache(self, prompt: List[int], prompt_cache: Optional[List[Any]] = None) -> List[int]:
        """Manage prompt caching."""
        # Use provided prompt_cache or instance-level prompt_cache
        active_prompt_cache = prompt_cache if prompt_cache is not None else self.prompt_cache

        # Debug logging to inspect inputs
        logger.debug(f"Type of prompt_cache: {type(prompt_cache)}")
        logger.debug(f"Type of self.prompt_cache: {type(self.prompt_cache)}")

        # Validate that active_prompt_cache is a list
        if not isinstance(active_prompt_cache, list):
            logger.error(
                f"Expected list for prompt_cache, got {type(active_prompt_cache).__name__}")
            raise TypeError(
                f"prompt_cache must be a list, got {type(active_prompt_cache).__name__}")

        # If cache is empty or model has changed, initialize new cache
        if not active_prompt_cache or self.model_provider.model_key != getattr(self, '_last_model_key', None):
            logger.debug("Initializing new prompt cache")
            active_prompt_cache.clear()
            active_prompt_cache.extend(
                make_prompt_cache(self.model_provider.model))
            if self.model_provider.draft_model is not None:
                active_prompt_cache.extend(
                    make_prompt_cache(self.model_provider.draft_model))
            setattr(self, '_last_model_key', self.model_provider.model_key)

        # Return full prompt (simplified, as list-based cache is managed by stream_generate)
        logger.debug(
            "Returning full prompt as cache is managed by stream_generate")
        return prompt

    def _generate_completion(
        self,
        prompt: List[int],
        model_obj: Any,
        tokenizer: MLXTokenizer,
        stop_id_sequences: List[List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        min_p: float,
        min_tokens_to_keep: int,
        top_k: int,
        repetition_penalty: Optional[float],
        repetition_context_size: int,
        xtc_probability: float,
        xtc_threshold: float,
        logit_bias: Optional[Dict[int, float]],
        logprobs: int,
        request_id: str,
        object_type: str,
        draft_model: Optional[Any],
        num_draft_tokens: int,
        verbose: bool = False,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Union[Literal["text", "json"],
                               JsonSchemaValue] = "text"
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Core method to generate non-streaming completions."""
        active_prompt_cache = prompt_cache if prompt_cache is not None else self.prompt_cache
        logit_bias = convert_logit_bias(logit_bias, tokenizer)
        if verbose:
            logger.newline()
            logger.info(f"Prompt: ({len(prompt)})")
            if logit_bias:
                logger.newline()
                logger.info("logit_bias:")
                logger.orange(format_json(logit_bias))
                for token in logit_bias.keys():
                    choice = tokenizer.decode(token)
                    logger.log("Token for", f"'{choice}'", ":", token, colors=[
                               "GRAY", "ORANGE", "GRAY", "ORANGE"])
        tokens: List[int] = []
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []
        text: str = ""
        finish_reason: Literal["length", "stop"] = "length"
        cached_prompt = self._get_prompt_cache(
            prompt, prompt_cache=active_prompt_cache)
        sampler = make_sampler(
            temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            min_tokens_to_keep=min_tokens_to_keep,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            xtc_special_tokens=[
                tokenizer.eos_token_id, tokenizer.encode("\n")],
        )
        logits_processors = make_logits_processors(
            logit_bias, repetition_penalty, repetition_context_size)
        stop_texts = tokenizer.batch_decode(stop_id_sequences)
        all_repetitions = []
        for gen_response in stream_generate(
            model=model_obj,
            tokenizer=tokenizer,
            prompt=cached_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=active_prompt_cache,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        ):
            segment: str = gen_response.text
            text += segment
            token: int = gen_response.token
            logprobs_data: mx.array = gen_response.logprobs
            tokens.append(token)
            finish_reason = gen_response.finish_reason
            if verbose:
                logger.log(segment, flush=True, colors=["TEAL"])
            if logprobs > 0:
                sorted_indices: mx.array = mx.argpartition(
                    -logprobs_data, kth=logprobs - 1)
                top_indices: mx.array = sorted_indices[:logprobs]
                top_logprobs: mx.array = logprobs_data[top_indices]
                top_token_info = zip(top_indices.tolist(),
                                     top_logprobs.tolist())
                top_tokens.append(dict(top_token_info))
            token_logprobs.append(logprobs_data[token].item())
            for stop_text in stop_texts:
                if stop_text in text:
                    finish_reason = "stop"
                    text = text[:text.index(stop_text)]
                    segment = segment[:segment.index(stop_text)]
                    break
            if finish_reason:
                logger.newline()
                break
        return self._generate_response(
            text=text,
            finish_reason=finish_reason,
            request_id=request_id,
            object_type=object_type,
            model=self.model_provider.model_key[0],
            prompt_token_count=gen_response.prompt_tokens,
            prompt_tps=gen_response.prompt_tps,
            completion_token_count=len(tokens),
            completion_tps=gen_response.generation_tps,
            peak_memory=gen_response.peak_memory,
            token_logprobs=token_logprobs,
            top_tokens=top_tokens,
            tokens=tokens,
            repetitions=all_repetitions,
            response_format=response_format
        )

    def _stream_generate_completion(
        self,
        prompt: List[int],
        model_obj: Any,
        tokenizer: MLXTokenizer,
        stop_id_sequences: List[List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        min_p: float,
        min_tokens_to_keep: int,
        top_k: int,
        repetition_penalty: Optional[float],
        repetition_context_size: int,
        xtc_probability: float,
        xtc_threshold: float,
        logit_bias: Optional[Dict[int, float]],
        logprobs: int,
        request_id: str,
        object_type: str,
        draft_model: Optional[Any],
        num_draft_tokens: int,
        verbose: bool = False,
        prompt_cache: Optional[List[Any]] = None,
        response_format: Union[Literal["text", "json"],
                               JsonSchemaValue] = "text"
    ) -> Iterator[CompletionResponse]:
        """Generate streaming completions."""
        active_prompt_cache = prompt_cache if prompt_cache is not None else self.prompt_cache
        logit_bias = convert_logit_bias(logit_bias, tokenizer)
        if verbose:
            logger.newline()
            logger.info(f"Prompt: ({len(prompt)})")
            if logit_bias:
                logger.newline()
                logger.info("logit_bias:")
                logger.orange(format_json(logit_bias))
                for token in logit_bias.keys():
                    choice = tokenizer.decode(token)
                    logger.log("Token for", f"'{choice}'", ":", token, colors=[
                               "GRAY", "ORANGE", "GRAY", "ORANGE"])
        tokens: List[int] = []
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []
        text: str = ""
        finish_reason: Optional[Literal["length", "stop"]] = None
        cached_prompt = self._get_prompt_cache(
            prompt, prompt_cache=active_prompt_cache)
        sampler = make_sampler(
            temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            min_tokens_to_keep=min_tokens_to_keep,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
            xtc_special_tokens=[
                tokenizer.eos_token_id, tokenizer.encode("\n")],
        )
        logits_processors = make_logits_processors(
            logit_bias, repetition_penalty, repetition_context_size)
        stop_texts = tokenizer.batch_decode(stop_id_sequences)
        all_repetitions = []
        for gen_response in stream_generate(
            model=model_obj,
            tokenizer=tokenizer,
            prompt=cached_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=active_prompt_cache,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        ):
            segment: str = gen_response.text
            text += segment
            token: int = gen_response.token
            logprobs_data: mx.array = gen_response.logprobs
            tokens.append(token)
            finish_reason = gen_response.finish_reason
            if verbose:
                logger.log(segment, flush=True, colors=["TEAL"])
            if logprobs > 0:
                sorted_indices: mx.array = mx.argpartition(
                    -logprobs_data, kth=logprobs - 1)
                top_indices: mx.array = sorted_indices[:logprobs]
                top_logprobs: mx.array = logprobs_data[top_indices]
                top_token_info = zip(top_indices.tolist(),
                                     top_logprobs.tolist())
                top_tokens.append(dict(top_token_info))
            token_logprobs.append(logprobs_data[token].item())
            for stop_text in stop_texts:
                if stop_text in text:
                    finish_reason = "stop"
                    text = text[:text.index(stop_text)]
                    segment = segment[:segment.index(stop_text)]
                    break
            yield self._generate_response(
                text=text,
                segment=segment,
                finish_reason=finish_reason,
                request_id=request_id,
                object_type=object_type,
                model=self.model_provider.model_key[0],
                prompt_token_count=gen_response.prompt_tokens,
                prompt_tps=gen_response.prompt_tps,
                completion_token_count=len(tokens),
                completion_tps=gen_response.generation_tps,
                peak_memory=gen_response.peak_memory,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                repetitions=all_repetitions,
                response_format=response_format
            )
            if finish_reason:
                logger.newline()
                break

    def reset_model(self) -> None:
        """Reset the model and clear associated caches."""
        logger.debug("\nResetting model and clearing caches")
        try:
            if hasattr(self, 'prompt_cache') and self.prompt_cache:
                # Clear the prompt cache
                self.prompt_cache.clear()
        except AttributeError:
            logger.warning(
                "prompt_cache attribute not found, skipping cache clear")
        self.model_provider = ModelProvider(self.cli_args)
        self.model = self.model_provider.model
        self.tokenizer = self.model_provider.tokenizer
        mx.reset_peak_memory()
        mx.clear_cache()
        logger.debug("Model reset completed")

    def print_cache(self) -> None:
        logger.debug("\nPrinting prompt cache details")
        if not self.prompt_cache:
            logger.info("Prompt cache is empty")
            return
        logger.info("Prompt Cache Details:")
        logger.info(f"  Cache size: {len(self.prompt_cache)} entries")
        total_memory = 0
        for i, cache_entry in enumerate(self.prompt_cache):
            logger.info(f"  Cache entry {i}:")
            logger.info(f"    Type: {type(cache_entry).__name__}")
            if isinstance(cache_entry, (list, tuple)):
                logger.info(f"    Length: {len(cache_entry)}")
            elif isinstance(cache_entry, KVCache):
                logger.info(f"    Offset: {cache_entry.offset}")
                logger.info(f"    Step: {cache_entry.step}")
                keys_shape = cache_entry.keys.shape if cache_entry.keys is not None else None
                values_shape = cache_entry.values.shape if cache_entry.values is not None else None
                logger.info(f"    Keys shape: {keys_shape or 'None'}")
                logger.info(f"    Values shape: {values_shape or 'None'}")
                if keys_shape and values_shape:
                    keys_memory = keys_shape[0] * keys_shape[1] * \
                        keys_shape[2] * keys_shape[3] * 4 / 1024 / 1024
                    values_memory = values_shape[0] * values_shape[1] * \
                        values_shape[2] * values_shape[3] * 4 / 1024 / 1024
                    logger.info(
                        f"    Memory usage: {keys_memory + values_memory:.2f} MB")
                    total_memory += keys_memory + values_memory
            elif hasattr(cache_entry, 'max_size'):
                logger.info(
                    f"    Max size: {getattr(cache_entry, 'max_size', 'N/A')}")
                logger.info(
                    f"    Keep: {getattr(cache_entry, 'keep', 'N/A')}")
            else:
                logger.info(
                    "    Contents: (Non-list/tuple cache, details not displayed)")
        logger.info(f"  Total memory usage: {total_memory:.2f} MB")
