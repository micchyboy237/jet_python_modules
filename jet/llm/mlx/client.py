import argparse
import uuid
import json
import time
from typing import Dict, List, Optional, Union, Literal, TypedDict, Any
from dataclasses import dataclass
from huggingface_hub import scan_cache_dir
import mlx.core as mx
from mlx_lm.server import ModelProvider, PromptCache, get_system_fingerprint, sequence_overlap, stopping_criteria, convert_chat, process_message_content
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import load

# Typed dictionaries for structured data


class Message(TypedDict):
    role: str
    content: str


class Delta(TypedDict):
    role: Optional[str]
    content: Optional[str]


class Tool(TypedDict):
    type: str
    function: Dict[str, Any]


class RoleMapping(TypedDict, total=False):
    system_prompt: str
    system: str
    user: str
    assistant: str
    stop: str


class Logprobs(TypedDict):
    token_logprobs: List[float]
    top_logprobs: List[Dict[int, float]]
    tokens: List[int]


class Choice(TypedDict):
    index: int
    logprobs: Logprobs
    finish_reason: Optional[Literal["length", "stop"]]
    message: Optional[Message]
    delta: Optional[Delta]
    text: Optional[str]


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(TypedDict):
    id: str
    system_fingerprint: str
    object: str
    model: str
    created: int
    choices: List[Choice]
    usage: Optional[Usage]


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


class MLXLMClient:
    """A client for interacting with MLX-LM models directly in Python."""

    @dataclass
    class Config:
        model: Optional[str] = None
        adapter_path: Optional[str] = None
        draft_model: Optional[str] = None
        trust_remote_code: bool = False
        chat_template: Optional[str] = None
        use_default_chat_template: bool = True

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the client with configuration."""
        if config is None:
            config = self.Config()

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

    def chat(
        self,
        messages: List[Message],
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        role_mapping: Optional[RoleMapping] = None,
        tools: Optional[List[Tool]] = None
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion."""
        # Validate parameters
        self._validate_parameters(
            stream, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model, adapter
        )

        # Load model
        model_obj: Any
        tokenizer: Any
        model_obj, tokenizer = self.model_provider.load(
            model, adapter, draft_model)

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"chatcmpl-{uuid.uuid4()}"
        object_type: str = "chat.completion.chunk" if stream else "chat.completion"
        if tokenizer.chat_template:
            process_message_content(messages)
            prompt: List[int] = tokenizer.apply_chat_template(
                messages, tools, add_generation_prompt=True
            )
        else:
            prompt_str: str = convert_chat(messages, role_mapping)
            prompt = tokenizer.encode(prompt_str)

        # Generate completion
        return self._generate_completion(
            prompt=prompt,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
            num_draft_tokens=3
        )

    def generate(
        self,
        prompt: str,
        model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        draft_model: Optional[str] = None,
        adapter: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        repetition_context_size: int = 20,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logprobs: int = -1,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a text completion."""
        # Validate parameters
        self._validate_parameters(
            stream, max_tokens, temperature, top_p, repetition_penalty,
            repetition_context_size, xtc_probability, xtc_threshold,
            logit_bias, logprobs, model, adapter
        )

        # Load model
        model_obj: Any
        tokenizer: Any
        model_obj, tokenizer = self.model_provider.load(
            model, adapter, draft_model)

        # Prepare stop sequences
        stop_words: List[str] = [stop] if isinstance(stop, str) else stop or []
        stop_id_sequences: List[List[int]] = [
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Generate prompt
        request_id: str = f"cmpl-{uuid.uuid4()}"
        object_type: str = "text_completion"
        prompt_tokens: List[int] = tokenizer.encode(prompt)

        # Generate completion
        return self._generate_completion(
            prompt=prompt_tokens,
            model_obj=model_obj,
            tokenizer=tokenizer,
            stop_id_sequences=stop_id_sequences,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
            num_draft_tokens=3
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
        repetition_penalty: float,
        repetition_context_size: int,
        xtc_probability: float,
        xtc_threshold: float,
        logit_bias: Optional[Dict[int, float]],
        logprobs: int,
        model: str,
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
        if not isinstance(repetition_penalty, (float, int)) or repetition_penalty < 0:
            raise ValueError("repetition_penalty must be a non-negative float")
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
        if not isinstance(model, str):
            raise ValueError("model must be a string")
        if adapter is not None and not isinstance(adapter, str):
            raise ValueError("adapter must be a string")
        if logit_bias is not None:
            if not isinstance(logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")
            try:
                logit_bias = {int(k): v for k, v in logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")

    def _generate_response(
        self,
        text: str,
        finish_reason: Optional[Literal["length", "stop"]],
        request_id: str,
        object_type: str,
        model: str,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
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
            "usage": None
        }

        if not stream:
            if not (isinstance(prompt_token_count, int) and isinstance(completion_token_count, int)):
                raise ValueError(
                    "Response type is complete, but token counts not provided")
            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]
        if object_type.startswith("chat.completion"):
            key_name = "delta" if stream else "message"
            choice[key_name] = {"role": "assistant", "content": text}
        elif object_type == "text_completion":
            choice["text"] = text
        else:
            raise ValueError(f"Unsupported response type: {object_type}")

        return response

    def _completion_usage_response(
        self,
        request_id: str,
        model: str,
        prompt_token_count: int,
        completion_token_count: int
    ) -> CompletionResponse:
        """Generate a usage-only response for streaming."""
        return {
            "id": request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }

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

    def _generate_completion(
        self,
        prompt: List[int],
        model_obj: Any,
        tokenizer: Any,
        stop_id_sequences: List[List[int]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        repetition_context_size: int,
        xtc_probability: float,
        xtc_threshold: float,
        logit_bias: Optional[Dict[int, float]],
        logprobs: int,
        stream: bool,
        request_id: str,
        object_type: str,
        draft_model: Optional[Any],
        num_draft_tokens: int
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Core method to generate completions."""
        tokens: List[int] = []
        token_logprobs: List[float] = []
        top_tokens: List[Dict[int, float]] = []
        text: str = ""
        finish_reason: Literal["length", "stop"] = "length"

        prompt = self._get_prompt_cache(prompt)
        sampler = make_sampler(
            temperature,
            top_p=top_p,
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

                if logprobs > 0:
                    sorted_indices: mx.array = mx.argpartition(
                        -logprobs_data, kth=logprobs - 1)
                    top_indices: mx.array = sorted_indices[:logprobs]
                    top_logprobs: mx.array = logprobs_data[top_indices]
                    top_token_info = zip(
                        top_indices.tolist(), top_logprobs.tolist())
                    top_tokens.append(dict(top_token_info))

                token_logprobs.append(logprobs_data[token].item())

                stop_condition = stopping_criteria(
                    tokens, stop_id_sequences, tokenizer.eos_token_id)
                if stop_condition.stop_met:
                    finish_reason = "stop"
                    if stop_condition.trim_length:
                        stop_sequence_suffix: str = tokenizer.decode(
                            tokens[-stop_condition.trim_length:])
                        text = text[:-len(stop_sequence_suffix)]
                    responses.append(self._generate_response(
                        text=text,
                        finish_reason=finish_reason,
                        request_id=request_id,
                        object_type=object_type,
                        model=self.model_provider.model_key[0],
                        stream=stream
                    ))
                    break

                if segment and not any(sequence_overlap(tokens, seq) for seq in stop_id_sequences):
                    responses.append(self._generate_response(
                        text=segment,
                        finish_reason=None,
                        request_id=request_id,
                        object_type=object_type,
                        model=self.model_provider.model_key[0],
                        stream=stream
                    ))

            self.prompt_cache.tokens.extend(tokens)
            responses.append(self._generate_response(
                text="",
                finish_reason=finish_reason,
                request_id=request_id,
                object_type=object_type,
                model=self.model_provider.model_key[0],
                stream=stream
            ))
            if stream and {"include_usage": True} in [self.cli_args.__dict__.get("stream_options", {})]:
                responses.append(self._completion_usage_response(
                    request_id=request_id,
                    model=self.model_provider.model_key[0],
                    prompt_token_count=len(prompt),
                    completion_token_count=len(tokens)
                ))
            return responses
        else:
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

                if logprobs > 0:
                    sorted_indices: mx.array = mx.argpartition(
                        -logprobs_data, kth=logprobs - 1)
                    top_indices: mx.array = sorted_indices[:logprobs]
                    top_logprobs: mx.array = logprobs_data[top_indices]
                    top_token_info = zip(
                        top_indices.tolist(), top_logprobs.tolist())
                    top_tokens.append(dict(top_token_info))

                token_logprobs.append(logprobs_data[token].item())

                stop_condition = stopping_criteria(
                    tokens, stop_id_sequences, tokenizer.eos_token_id)
                if stop_condition.stop_met:
                    finish_reason = "stop"
                    if stop_condition.trim_length:
                        stop_sequence_suffix: str = tokenizer.decode(
                            tokens[-stop_condition.trim_length:])
                        text = text[:-len(stop_sequence_suffix)]
                    break

            self.prompt_cache.tokens.extend(tokens)
            return self._generate_response(
                text=text,
                finish_reason=finish_reason,
                request_id=request_id,
                object_type=object_type,
                model=self.model_provider.model_key[0],
                prompt_token_count=len(prompt),
                completion_token_count=len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                stream=stream
            )
