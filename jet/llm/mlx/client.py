import argparse
import os
import uuid
import json
import time
import jet.llm.mlx.model_cache  # Activates cleanup listener
from typing import Dict, List, Optional, Union, Literal, TypedDict, Any, Iterator
from dataclasses import dataclass
from huggingface_hub import scan_cache_dir
from jet.llm.mlx.config import DEFAULT_MODEL
from jet.logger import logger
from jet.transformers.formatters import format_json
import mlx.core as mx
from mlx_lm.server import ModelProvider, PromptCache, get_system_fingerprint, sequence_overlap, stopping_criteria, convert_chat, process_message_content
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import load
from jet.llm.mlx.utils.logit_bias import convert_logit_bias
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.mlx_types import (
    MLXTokenizer,
    Message,
    Tool,
    RoleMapping,
    CompletionResponse,
    ModelsResponse,
    ModelInfo,
    LLMModelType,
    ModelKey,
    ModelValue,
)
from jet.llm.mlx.models import AVAILABLE_MODELS, resolve_model_value
from jet.utils.inspect_utils import get_entry_file_name

DEFAULT_LOG_DIR = os.path.expanduser(
    f"~/.cache/mlx-logs/{get_entry_file_name()}")


class MLXLMClient:
    """A client for interacting with MLX-LM models directly in Python."""

    @staticmethod
    def _get_model_value(model: LLMModelType) -> ModelValue:
        """Convert a model key to its full value if it exists in AVAILABLE_MODELS."""
        if not isinstance(model, str):
            raise ValueError("Model must be a string (ModelKey or ModelValue)")

        # Check if the model is a valid ModelValue (full path)
        if model in AVAILABLE_MODELS.values():
            return model  # type: ignore

        # Check if the model is a valid ModelKey (short name)
        if model in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model]

        # If not found, return the input as is (assuming it's a valid ModelValue)
        return model  # type: ignore

    @dataclass
    class Config:
        model: Optional[LLMModelType] = None
        adapter_path: Optional[str] = None
        draft_model: Optional[LLMModelType] = None
        trust_remote_code: bool = False
        chat_template: Optional[str] = None
        use_default_chat_template: bool = True

    def __init__(
        self,
        model: Optional[LLMModelType] = None,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        seed: Optional[int] = None,
        log_dir: str = DEFAULT_LOG_DIR,
    ) -> None:
        """Initialize the client with configuration."""
        if seed:
            mx.random.seed(seed)

        # Convert model keys to values
        model_value = resolve_model_value(model) if model else None
        draft_model_value = resolve_model_value(
            draft_model) if draft_model else None

        config = self.Config(
            model=model_value,
            adapter_path=adapter_path,
            draft_model=draft_model_value,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template
        )

        # Create CLI args equivalent
        self.cli_args: argparse.Namespace = argparse.Namespace(
            model=config.model,
            adapter_path=config.adapter_path,
            draft_model=config.draft_model,
            trust_remote_code=config.trust_remote_code,
            chat_template=config.chat_template,
            use_default_chat_template=config.use_default_chat_template
        )

        # Initialize model provider and other attributes
        self.model_provider: ModelProvider = ModelProvider(self.cli_args)
        self.prompt_cache: PromptCache = PromptCache()
        self.system_fingerprint: str = get_system_fingerprint()
        self.created: int = int(time.time())
        self.log_dir = log_dir

        self.model = self.model_provider.model
        self.tokenizer: MLXTokenizer = self.model_provider.tokenizer

    def chat(
        self,
        messages: List[Message],
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        top_k: int = 0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion."""
        # Convert model keys to values
        model_value = self._get_model_value(model)
        draft_model_value = self._get_model_value(
            draft_model) if draft_model else None

        # Validate parameters
        self._validate_parameters(
            stream, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model_value, adapter
        )

        # Load model
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)

        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"chatcmpl-{uuid.uuid4()}"
        object_type: str = "stream.chat.completion" if stream else "chat.completion"
        if tokenizer.chat_template:
            process_message_content(messages)
            prompt: List[int] = tokenizer.apply_chat_template(
                messages,
                tools,
                add_generation_prompt=True,
                # Switches between thinking and non-thinking modes. Default is True.
                enable_thinking=False,
            )
        else:
            prompt_str: str = convert_chat(messages, role_mapping)
            prompt = tokenizer.encode(prompt_str)

        # Generate completion
        response = self._generate_completion(
            prompt=prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
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
            stream=stream,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=verbose
        )

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="chat").log_interaction(
                messages,
                response,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                logprobs=logprobs,
                stream=stream
            )
        return response

    def stream_chat(
        self,
        messages: List[Message],
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
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
        verbose: bool = False
    ) -> Iterator[CompletionResponse]:
        """Stream chat completions as they are generated."""
        # Convert model keys to values
        model_value = self._get_model_value(model)
        draft_model_value = self._get_model_value(
            draft_model) if draft_model else None

        # Validate parameters
        self._validate_parameters(
            True, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model_value, adapter
        )

        # Load model
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)

        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"chatcmpl-{uuid.uuid4()}"
        object_type: str = "stream.chat.completion"
        if tokenizer.chat_template:
            process_message_content(messages)
            prompt: List[int] = tokenizer.apply_chat_template(
                messages,
                tools,
                add_generation_prompt=True,
                # Switches between thinking and non-thinking modes. Default is True.
                enable_thinking=False,
            )
        else:
            prompt_str: str = convert_chat(messages, role_mapping)
            prompt = tokenizer.encode(prompt_str)

        # Stream completion
        for response in self._stream_generate_completion(
            prompt=prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
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
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=verbose
        ):
            yield response

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_chat").log_interaction(
                messages,
                response,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                logprobs=logprobs,
                stream=True
            )

    def generate(
        self,
        prompt: str,
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        top_k: int = 0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Union[Dict[int, float],
                                   Dict[str, float], str, List[str]]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        log_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a text completion."""
        # Convert model keys to values
        model_value = self._get_model_value(model)
        draft_model_value = self._get_model_value(
            draft_model) if draft_model else None

        # Validate parameters
        self._validate_parameters(
            stream, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model_value, adapter
        )

        # Load model
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)

        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"cmpl-{uuid.uuid4()}"
        object_type: str = "text.completion"
        prompt_tokens: List[int] = tokenizer.encode(prompt)

        # Generate completion
        response = self._generate_completion(
            prompt=prompt_tokens,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
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
            stream=stream,
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=verbose
        )

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="generate").log_interaction(
                prompt,
                response,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                logprobs=logprobs,
                stream=stream
            )
        return response

    def stream_generate(
        self,
        prompt: str,
        model: LLMModelType = DEFAULT_MODEL,
        draft_model: Optional[LLMModelType] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
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
        verbose: bool = False
    ) -> Iterator[CompletionResponse]:
        """Stream text completions as they are generated."""
        # Convert model keys to values
        model_value = self._get_model_value(model)
        draft_model_value = self._get_model_value(
            draft_model) if draft_model else None

        # Validate parameters
        self._validate_parameters(
            True, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model_value, adapter
        )

        # Load model
        model_obj, tokenizer = self.model_provider.load(
            model_value, adapter, draft_model_value)

        if tokenizer is None:
            raise ValueError("Failed to load tokenizer")

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"cmpl-{uuid.uuid4()}"
        object_type: str = "stream.text.completion"
        prompt_tokens: List[int] = tokenizer.encode(prompt)

        # Stream completion
        for response in self._stream_generate_completion(
            prompt=prompt_tokens,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
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
            request_id=request_id,
            object_type=object_type,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=3,
            verbose=verbose
        ):
            yield response

        # Log interaction
        log_dir = log_dir or self.log_dir
        if log_dir:
            ChatLogger(log_dir, method="stream_generate").log_interaction(
                prompt,
                response,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                logprobs=logprobs,
                stream=True
            )

    def get_models(self) -> ModelsResponse:
        """List available models."""
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

        # Create a list of available models
        models: List[ModelInfo] = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        return {"object": "list", "data": models}

    def _validate_parameters(
        self,
        stream: bool,
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
        adapter: Optional[str]
    ) -> None:
        """Validate model parameters."""
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        if not isinstance(max_tokens, int) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
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
        if not isinstance(model, (str, ModelKey, ModelValue)):
            raise ValueError("model must be a string or valid model type")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError("adapter must be a string")
        if logit_bias is not None:
            if not isinstance(logit_bias, (dict, str, list)):
                raise ValueError(
                    "logit_bias must be a dict of int to float, str or list[str]")

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
        stream: bool = False
    ) -> CompletionResponse:
        """Generate a response packet in OpenAI-compatible format."""
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tokens = tokens or []

        response: CompletionResponse = {
            "id": request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": object_type,
            "model": model,
            "created": self.created,
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
            "usage": None,
            "content": text
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
            # key_name = "delta" if stream else "message"
            choice["message"] = {"role": "assistant",
                                 "content": segment if segment != None else text}
        elif "text.completion" in object_type:
            choice["text"] = segment if segment != None else text
        else:
            raise ValueError(f"Unsupported response type: {object_type}")

        return response

    def _get_prompt_cache(self, prompt: List[int]) -> List[int]:
        """Manage prompt caching."""
        cache_len: int = len(self.prompt_cache.tokens)
        prompt_len: int = len(prompt)
        prefix_len: int = min(cache_len, prompt_len)

        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or prompt[:prefix_len] != self.prompt_cache.tokens[:prefix_len]
        ):
            self.prompt_cache.model_key = self.model_provider.model_key
            self.prompt_cache.cache = make_prompt_cache(
                self.model_provider.model)
            if self.model_provider.draft_model is not None:
                self.prompt_cache.cache += make_prompt_cache(
                    self.model_provider.draft_model)
            self.prompt_cache.tokens = []
        elif cache_len >= prompt_len:
            if can_trim_prompt_cache(self.prompt_cache.cache):
                num_to_trim: int = cache_len - prompt_len + 1
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:-num_to_trim]
                prompt = prompt[-1:]
            else:
                self.prompt_cache.cache = make_prompt_cache(
                    self.model_provider.model)
                if self.model_provider.draft_model is not None:
                    self.prompt_cache.cache += make_prompt_cache(
                        self.model_provider.draft_model)
                self.prompt_cache.tokens = []
        else:
            prompt = prompt[cache_len:]

        self.prompt_cache.tokens.extend(prompt)
        return prompt

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
        verbose: bool = False
    ) -> Iterator[CompletionResponse]:
        """Generate streaming completions."""

        # Handle logit_bias conversion
        logit_bias = convert_logit_bias(logit_bias, tokenizer)

        if verbose:
            logger.newline()
            logger.info("Prompt:")
            logger.debug(tokenizer.decode(prompt))

            if logit_bias:
                logger.newline()
                logger.info("logit_bias:")
                logger.orange(format_json(logit_bias))
                for token in logit_bias.keys():
                    choice = tokenizer.decode(token)
                    logger.log("Token for", f"'{choice}'", ":",
                               token, colors=["GRAY", "ORANGE", "GRAY" "ORANGE"])

        tokens: List[int] = []
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []
        text: str = ""
        finish_reason: Optional[Literal["length", "stop"]] = None

        prompt = self._get_prompt_cache(prompt)
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
            logit_bias, repetition_penalty, repetition_context_size
        )

        stop_texts = tokenizer.batch_decode(stop_id_sequences)
        for gen_response in stream_generate(
            model=model_obj,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prompt_cache=self.prompt_cache.cache,
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
                logger.success(segment, flush=True)

            if logprobs > 0:
                sorted_indices: mx.array = mx.argpartition(
                    -logprobs_data, kth=logprobs - 1)
                top_indices: mx.array = sorted_indices[:logprobs]
                top_logprobs: mx.array = logprobs_data[top_indices]
                top_token_info = zip(
                    top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(dict(top_token_info))

            token_logprobs.append(logprobs_data[token].item())

            # Check for stop texts in the generated text
            for stop_text in stop_texts:
                if stop_text in text:
                    finish_reason = "stop"
                    # Trim text up to stop_text
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
                stream=True,
                prompt_token_count=gen_response.prompt_tokens,
                prompt_tps=gen_response.prompt_tps,
                completion_token_count=len(tokens),
                completion_tps=gen_response.generation_tps,
                peak_memory=gen_response.peak_memory,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
            )

            if finish_reason:
                break

        self.prompt_cache.tokens.extend(tokens)

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
        stream: bool,
        request_id: str,
        object_type: str,
        draft_model: Optional[Any],
        num_draft_tokens: int,
        verbose: bool = False
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Core method to generate non-streaming completions."""

        # Handle logit_bias conversion
        logit_bias = convert_logit_bias(logit_bias, tokenizer)

        if verbose:
            logger.newline()
            logger.info("Prompt:")
            logger.debug(tokenizer.decode(prompt))

            if logit_bias:
                logger.newline()
                logger.info("logit_bias:")
                logger.orange(format_json(logit_bias))
                for token in logit_bias.keys():
                    choice = tokenizer.decode(token)
                    logger.log("Token for", f"'{choice}'", ":",
                               token, colors=["GRAY", "ORANGE", "GRAY" "ORANGE"])

        tokens: List[int] = []
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []
        text: str = ""
        finish_reason: Literal["length", "stop"] = "length"

        prompt = self._get_prompt_cache(prompt)
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
            logit_bias, repetition_penalty, repetition_context_size
        )

        if stream:
            responses: List[CompletionResponse] = []
            stop_texts = tokenizer.batch_decode(stop_id_sequences)
            for gen_response in stream_generate(
                model=model_obj,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self.prompt_cache.cache,
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
                    logger.success(segment, flush=True)

                if logprobs > 0:
                    sorted_indices: mx.array = mx.argpartition(
                        -logprobs_data, kth=logprobs - 1)
                    top_indices: mx.array = sorted_indices[:logprobs]
                    top_logprobs: mx.array = logprobs_data[top_indices]
                    top_token_info = zip(
                        top_indices.tolist(), top_logprobs.tolist())
                    top_tokens.append(dict(top_token_info))

                token_logprobs.append(logprobs_data[token].item())

                # Check for stop texts in the generated text
                for stop_text in stop_texts:
                    if stop_text in text:
                        # Trim text up to stop_text
                        text = text[:text.index(stop_text)]
                        segment = segment[:segment.index(stop_text)]
                        break

                responses.append(self._generate_response(
                    text="",
                    finish_reason=finish_reason,
                    request_id=request_id,
                    object_type=object_type,
                    model=self.model_provider.model_key[0],
                    stream=True,
                    prompt_token_count=gen_response.prompt_tokens,
                    prompt_tps=gen_response.prompt_tps,
                    completion_token_count=len(tokens),
                    completion_tps=gen_response.generation_tps,
                    peak_memory=gen_response.peak_memory,
                    token_logprobs=token_logprobs,
                    top_tokens=top_tokens,
                    tokens=tokens,
                ))

                if finish_reason:
                    break

            self.prompt_cache.tokens.extend(tokens)
            return responses
        else:
            stop_texts = tokenizer.batch_decode(stop_id_sequences)
            for gen_response in stream_generate(
                model=model_obj,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self.prompt_cache.cache,
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
                    logger.success(segment, flush=True)

                if logprobs > 0:
                    sorted_indices: mx.array = mx.argpartition(
                        -logprobs_data, kth=logprobs - 1)
                    top_indices: mx.array = sorted_indices[:logprobs]
                    top_logprobs: mx.array = logprobs_data[top_indices]
                    top_token_info = zip(
                        top_indices.tolist(), top_logprobs.tolist())
                    top_tokens.append(dict(top_token_info))

                token_logprobs.append(logprobs_data[token].item())

                # Check for stop texts in the generated text
                for stop_text in stop_texts:
                    if stop_text in text:
                        # Trim text up to stop_text
                        text = text[:text.index(stop_text)]
                        segment = segment[:segment.index(stop_text)]
                        break

                if finish_reason:
                    break

            self.prompt_cache.tokens.extend(tokens)
            return self._generate_response(
                text=text,
                finish_reason=finish_reason,
                request_id=request_id,
                object_type=object_type,
                model=self.model_provider.model_key[0],
                stream=True,
                prompt_token_count=gen_response.prompt_tokens,
                prompt_tps=gen_response.prompt_tps,
                completion_token_count=len(tokens),
                completion_tps=gen_response.generation_tps,
                peak_memory=gen_response.peak_memory,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
            )
