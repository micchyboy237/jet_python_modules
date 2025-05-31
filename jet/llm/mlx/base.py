from typing import Dict, List, Optional, Union, Iterator
from jet.logger import logger

from jet.llm.mlx.config import DEFAULT_MODEL
from jet.llm.mlx.mlx_types import MLXTokenizer, ModelKey, LLMModelType
from jet.llm.mlx.models import resolve_model
from jet.llm.mlx.utils.base import get_model_max_tokens
from jet.llm.mlx.token_utils import count_tokens, get_tokenizer_fn, merge_texts
from jet.llm.mlx.client import MLXLMClient, ModelsResponse, CompletionResponse, Message, RoleMapping, Tool
from jet.llm.mlx.chat_history import ChatHistory


# Typed dictionaries for structured data (reused from MLXLMClient for consistency)


class MLX:
    """Wrapper class for MLXLMClient with chat history management."""

    def __init__(
        self,
        # Model Config
        model: LLMModelType = DEFAULT_MODEL,
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        # DB Config
        dbname: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        session_id: Optional[str] = None,
        with_history: bool = False,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        """Initialize the MLX client with configuration and optional database."""
        self.model_path = resolve_model(model)
        self.with_history = with_history  # Store the with_history flag
        self.log_dir = log_dir
        # Initialize MLXLMClient
        self.client = MLXLMClient(
            model=model,
            adapter_path=adapter_path,
            draft_model=draft_model,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            use_default_chat_template=use_default_chat_template,
            seed=seed,
        )
        self.prompt_cache = self.client.prompt_cache
        self.system_fingerprint = self.client.system_fingerprint
        self.created = self.client.created
        self.log_dir = self.client.log_dir
        self.model = self.client.model
        self.tokenizer: MLXTokenizer = self.client.tokenizer

        # Set padding token if not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize chat history
        if with_history and dbname:
            self.history = ChatHistory(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                session_id=session_id
            )
        else:
            self.history = ChatHistory()

    def get_models(self) -> ModelsResponse:
        return self.client.get_models()

    def chat(
        self,
        messages: Union[str, List[Message]],
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
        system_prompt: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Union[CompletionResponse, List[CompletionResponse]]:
        """Generate a chat completion with history management."""

        # Prepare messages with history
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            if self.with_history:
                self.history.add_message("system", system_prompt)

        # Handle messages input: str or List[Message]
        if isinstance(messages, str):
            if self.with_history:
                self.history.add_message("user", messages)
        elif isinstance(messages, list):
            for msg in messages:
                if "role" in msg and "content" in msg:
                    if self.with_history:
                        self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "messages must be a string or a list of Message dictionaries")

        all_messages = self.history.get_messages() if self.with_history else (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + ([{"role": "user", "content": messages}] if isinstance(messages, str) else messages)

        if max_tokens == -1:
            # Set remaining tokens as max tokens
            max_tokens = self.get_remaining_tokens(all_messages)

        # Call MLXLMClient.chat
        response = self.client.chat(
            messages=all_messages,
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
            stream=stream,
            role_mapping=role_mapping,
            tools=tools,
            log_dir=log_dir,
            verbose=verbose,
        )

        # Add assistant response to history
        if self.with_history and isinstance(response, dict) and response.get("choices"):
            assistant_content = response["choices"][0].get(
                "message", {}).get("content", "")
            if assistant_content:
                self.history.add_message("assistant", assistant_content)

        return response

    def stream_chat(
        self,
        messages: Union[str, List[Message]],
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
        system_prompt: Optional[str] = None,
        log_dir: Optional[str] = None,
        verbose: bool = False
    ) -> Iterator[Union[CompletionResponse, List[CompletionResponse]]]:
        """Stream chat completions with history management."""
        # Prepare messages with history
        if system_prompt and not any(msg["role"] == "system" for msg in self.history.get_messages()):
            if self.with_history:
                self.history.add_message("system", system_prompt)

        # Handle messages input: str or List[Message]
        if isinstance(messages, str):
            if self.with_history:
                self.history.add_message("user", messages)
        elif isinstance(messages, list):
            for msg in messages:
                if "role" in msg and "content" in msg:
                    if self.with_history:
                        self.history.add_message(msg["role"], msg["content"])
                else:
                    raise ValueError(
                        "Each message in the list must have 'role' and 'content' keys")
        else:
            raise TypeError(
                "messages must be a string or a list of Message dictionaries")

        all_messages = self.history.get_messages() if self.with_history else (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        ) + ([{"role": "user", "content": messages}] if isinstance(messages, str) else messages)

        if max_tokens == -1:
            # Set remaining tokens as max tokens
            max_tokens = self.get_remaining_tokens(all_messages)

        # Stream responses
        assistant_content = ""
        for response in self.client.stream_chat(
            messages=all_messages,
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
        ):
            if response.get("choices"):
                content = response["choices"][0].get(
                    "message", {}).get("content", "")
                assistant_content += content
            yield response

        # Add assistant response to history
        if self.with_history and assistant_content:
            self.history.add_message("assistant", assistant_content)

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
    ) -> CompletionResponse:
        """Generate a text completion (no history)."""

        response = self.client.generate(
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
            stream=stream,
            log_dir=log_dir,
            verbose=verbose,
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
        """Stream text completions (no history)."""
        for response in self.client.stream_generate(
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
        ):
            yield response

    def clear_history(self):
        """Clear the chat history."""
        self.history.clear()

    def count_tokens(self, messages: str | List[str] | List[Dict], prevent_total: bool = False) -> int | list[int]:
        return count_tokens(self.model_path, messages, prevent_total)

    def filter_docs(self, messages: str | List[str] | List[Message], chunk_size: int, buffer: int = 1024) -> list[str]:
        """Filter documents to fit within model token limits."""
        # Convert messages to a single string
        if isinstance(messages, str):
            context = messages
        elif isinstance(messages, list):
            if all(isinstance(msg, str) for msg in messages):
                context = "\n\n".join(messages)
            elif all(isinstance(msg, dict) and "content" in msg for msg in messages):
                context = "\n\n".join(msg["content"] for msg in messages)
            else:
                raise ValueError(
                    "Messages list must contain strings or Message dictionaries")
        else:
            raise TypeError(
                "Messages must be a string or list of strings/dictionaries")

        # Get model max tokens and reserve buffer
        model_max_tokens = get_model_max_tokens(self.model_path)
        max_tokens = model_max_tokens - buffer

        # Merge texts to fit within token limit
        merged_texts = merge_texts(
            context, self.tokenizer, max_length=chunk_size)

        # Build filtered context
        filtered_contexts = []
        current_token_count = 0

        for text, token_count in zip(merged_texts["texts"], merged_texts["token_counts"]):
            if current_token_count + token_count > max_tokens:
                break
            filtered_contexts.append(text)
            current_token_count += token_count

        return filtered_contexts

    def get_remaining_tokens(self, messages: str | List[str] | List[Message]) -> int:
        model_max_tokens = get_model_max_tokens(self.model_path)
        prompt_tokens = self.count_tokens(messages)
        # Set remaining tokens as max tokens
        remaining_tokens = model_max_tokens - prompt_tokens
        return remaining_tokens
