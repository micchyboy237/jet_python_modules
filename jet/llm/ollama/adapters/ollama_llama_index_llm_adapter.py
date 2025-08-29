import json
import hashlib
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from contextvars import copy_context
from functools import wraps
from ollama import AsyncClient, Client
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms.llm import ToolSelection, Model
from llama_index.core.program.utils import process_streaming_objects, FlexibleModel
from llama_index.core.prompts import PromptTemplate
from llama_index.core.types import PydanticProgramMode
from llama_index.core.base.llms.generic_utils import (
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
)
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    ImageBlock,
    LLMMetadata,
    MessageRole,
    TextBlock,
)

from jet.token.token_utils import token_counter
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger


if TYPE_CHECKING:
    from llama_index.core.tools.types import BaseTool

DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_CONTEXT_WINDOW = 2048
DEFAULT_NUM_OUTPUTS = 256
DEFAULT_TEMPERATURE = 0.1
dispatcher = get_dispatcher(__name__)


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    return {k: v for k, v in response.items() if k not in exclude}


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = response.message.additional_kwargs.get("tool_calls", []) or []
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]

# Helper to preserve context in async calls


def preserve_context_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = copy_context()  # Capture the current context
        # Run function in copied context
        return await ctx.run(func, *args, **kwargs)
    return wrapper


class OllamaFunctionCallingAdapter(FunctionCallingLLM):
    """
    Ollama LLM with dynamic context window based on input token count, including tools.

    Visit https://ollama.com/ to download and install Ollama.

    Run `ollama serve` to start a server.

    Run `ollama pull <name>` to download a model to run.

    Examples:
        `pip install llama-index-llms-ollama`

        ```python
        from llama_index.llms.ollama import Ollama

        llm = Ollama(model="llama3.2", request_timeout=60.0)

        response = llm.complete("What is the capital of France?")
        print(response)
        ```

    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
    model: str = Field(description="The Ollama model to use.")
    temperature: Optional[float] = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
    )
    context_window: int = Field(
        default=2048,
        description="The maximum number of context tokens for the model, dynamically adjusted based on input.",
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    json_mode: bool = Field(
        default=False,
        description="Whether to use JSON mode for the Ollama API.",
    )
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {"num_gpu": 28},
        description="Additional model parameters for the Ollama API, including GPU offloading.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model is a function calling model.",
    )
    keep_alive: Optional[Union[float, str]] = Field(
        default="5m",
        description="Controls how long the model will stay loaded into memory following the request (default: 5m)",
    )
    thinking: Optional[bool] = Field(
        default=None,
        description="Whether to enable or disable thinking in the model.",
    )
    log_dir: Optional[str] = Field(
        default=None,
        description="Directory path for saving log files.",
    )

    _client: Optional[Client] = PrivateAttr()
    _async_client: Optional[AsyncClient] = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        request_timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
        prompt_key: str = "prompt",
        json_mode: bool = False,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
        is_function_calling_model: bool = True,
        keep_alive: Optional[Union[float, str]] = None,
        thinking: Optional[bool] = None,
        log_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            context_window=context_window,
            request_timeout=request_timeout,
            prompt_key=prompt_key,
            json_mode=json_mode,
            additional_kwargs={**{"num_gpu": 28}, **(additional_kwargs or {})},
            is_function_calling_model=is_function_calling_model,
            keep_alive=keep_alive,
            thinking=thinking,
            **kwargs,
        )

        self._client = client
        self._async_client = async_client
        self.log_dir = log_dir

    @classmethod
    def class_name(cls) -> str:
        return "Ollama_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.get_context_window(),
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
        )

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(host=self.base_url,
                                  timeout=self.request_timeout)
        return self._client

    @property
    def async_client(self) -> AsyncClient:
        if self._async_client is None:
            self._async_client = AsyncClient(
                host=self.base_url, timeout=self.request_timeout)
        return self._async_client

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {**base_kwargs, **self.additional_kwargs}

    def get_context_window(self) -> int:
        if self.context_window == -1:
            info = self.client.show(self.model).modelinfo
            for key, value in info.items():
                if "context_length" in key:
                    self.context_window = int(value)
                    break
        return self.context_window if self.context_window != -1 else 2048

    def _get_dynamic_context_window(self, messages: Sequence[ChatMessage], tools: Optional[List[Any]] = None) -> int:
        """Calculate dynamic context window based on input token count, including messages and tools."""
        token_count = 0
        if messages:
            message_content = [msg.content for msg in messages if msg.content]
            token_count += token_counter(message_content, model=self.model)
        if tools:
            tool_specs = []
            for tool in tools:
                tool_specs.append(tool)
            tool_json = json.dumps(tool_specs)
            token_count += token_counter(tool_json, model=self.model)
        token_count += 50
        return min(int(token_count * 1.2), 2048)

    def _convert_to_ollama_messages(self, messages: Sequence[ChatMessage]) -> Dict:
        ollama_messages = []
        for message in messages:
            cur_ollama_message = {
                "role": message.role.value,
                "content": "",
            }
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    cur_ollama_message["content"] += block.text
                elif isinstance(block, ImageBlock):
                    if "images" not in cur_ollama_message:
                        cur_ollama_message["images"] = []
                    cur_ollama_message["images"].append(
                        block.resolve_image(
                            as_base64=True).read().decode("utf-8")
                    )
                else:
                    raise ValueError(f"Unsupported block type: {type(block)}")
            if "tool_calls" in message.additional_kwargs:
                cur_ollama_message["tool_calls"] = message.additional_kwargs["tool_calls"]
            ollama_messages.append(cur_ollama_message)
        return ollama_messages

    def _get_response_token_counts(self, raw_response: dict) -> dict:
        """Get the token usage reported by the response."""
        try:
            prompt_tokens = raw_response["prompt_eval_count"]
            completion_tokens = raw_response["eval_count"]
            total_tokens = prompt_tokens + completion_tokens
        except KeyError:
            return {}
        except TypeError:
            return {}
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _prepare_chat_with_tools(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        tool_specs = [tool.metadata.to_openai_tool(
            skip_length_check=True) for tool in tools]
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)
        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)
        return {"messages": messages, "tools": tool_specs or None}

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate the response from chat_with_tools."""
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolSelection]:
        """Predict and call the tool."""
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1 and error_on_no_tool_call:
            raise ValueError(
                f"Expected at least one tool call, but got {len(tool_calls)} tool calls.")
        tool_selections = []
        for tool_call in tool_calls:
            argument_dict = tool_call["function"]["arguments"]
            tool_selections.append(
                ToolSelection(
                    tool_id=tool_call["function"]["name"],
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=argument_dict,
                )
            )
        return tool_selections

    def _save_logs(self, request: Dict[str, Any], response: Dict[str, Any]) -> None:
        """Save request and response logs to a file if log_dir is specified."""
        if not self.log_dir:
            return
        # Serialize request and response to JSON for consistent hashing
        content = format_json({"request": request, "response": response})
        # Compute MD5 hash of the content
        # Use first 8 chars for brevity
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        timestamp = int(time.time())
        # Use first 8 chars of UUID for brevity
        unique_id = str(uuid.uuid4())[:8]
        log_file_name = f"{timestamp}_{content_hash}_{unique_id}.log"
        log_file_path = f"{self.log_dir}/{log_file_name}"
        _logger = CustomLogger(log_file_path, name=log_file_name)
        _logger.orange(f"Logs: {log_file_path}")
        _logger.pretty({
            "request": request,
            "response": response
        })

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ollama_messages = self._convert_to_ollama_messages(messages)
        tools = kwargs.pop("tools", None)
        dynamic_context = self._get_dynamic_context_window(messages, tools)
        options = {**self._model_kwargs, "num_ctx": dynamic_context}
        think = kwargs.pop("think", None) or self.thinking
        format = kwargs.pop("format", "json" if self.json_mode else None)

        request = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "format": format,
            "tools": tools,
            "think": think,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        response = self.client.chat(**request)
        response = dict(response)

        tool_calls = response["message"].get("tool_calls", [])
        thinking = response["message"].get("thinking", None)
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts

        self._save_logs(request, response)

        return ChatResponse(
            message=ChatMessage(
                content=response["message"].get("content", ""),
                role=response["message"].get("role", MessageRole.ASSISTANT),
                additional_kwargs={
                    "tool_calls": tool_calls, "thinking": thinking},
            ),
            raw=response,
        )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        ollama_messages = self._convert_to_ollama_messages(messages)
        tools = kwargs.pop("tools", None)
        dynamic_context = self._get_dynamic_context_window(messages, tools)
        options = {**self._model_kwargs, "num_ctx": dynamic_context}
        think = kwargs.pop("think", None) or self.thinking
        format = kwargs.pop("format", "json" if self.json_mode else None)

        request = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "format": format,
            "tools": tools,
            "think": think,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        def gen() -> ChatResponseGen:
            response = self.client.chat(**request)
            response_txt = ""
            thinking_txt = ""
            seen_tool_calls = set()
            all_tool_calls = []
            final_response = {}  # Store final response for logging

            for r in response:
                if r["message"]["content"] is None:
                    continue
                r = dict(r)
                response_txt += r["message"].get("content", "") or ""
                thinking_txt += r["message"].get("thinking", "") or ""
                new_tool_calls = [dict(t)
                                  for t in r["message"].get("tool_calls") or []]
                for tool_call in new_tool_calls:
                    tool_key = (str(tool_call["function"]["name"]), str(
                        tool_call["function"]["arguments"]))
                    if tool_key in seen_tool_calls:
                        continue
                    seen_tool_calls.add(tool_key)
                    all_tool_calls.append(tool_call)
                token_counts = self._get_response_token_counts(r)
                if token_counts:
                    r["usage"] = token_counts
                final_response = r
                # Safely set response_txt in final_response["message"]["content"]
                if "message" not in final_response:
                    final_response["message"] = {}
                final_response["message"]["content"] = response_txt
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"].get("role", MessageRole.ASSISTANT),
                        additional_kwargs={"tool_calls": list(
                            set(all_tool_calls)), "thinking": thinking_txt},
                    ),
                    delta=r["message"].get("content", ""),
                    raw=r,
                    additional_kwargs={
                        "thinking_delta": r["message"].get("thinking", None)},
                )

            # Log the final aggregated response
            if final_response:
                self._save_logs(request, final_response)

        return gen()

    @llm_chat_callback()
    @preserve_context_async
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        ollama_messages = self._convert_to_ollama_messages(messages)
        tools = kwargs.pop("tools", None)
        dynamic_context = self._get_dynamic_context_window(messages, tools)
        options = {**self._model_kwargs, "num_ctx": dynamic_context}
        think = kwargs.pop("think", None) or self.thinking
        format = kwargs.pop("format", "json" if self.json_mode else None)
        request = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "format": format,
            "tools": tools,
            "think": think,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        response = await self.async_client.chat(**request)
        response = dict(response)
        tool_calls = response["message"].get("tool_calls", [])
        thinking = response["message"].get("thinking", None)
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts
        self._save_logs(request, response)
        return ChatResponse(
            message=ChatMessage(
                content=response["message"].get("content", ""),
                role=response["message"].get("role", MessageRole.ASSISTANT),
                additional_kwargs={
                    "tool_calls": tool_calls, "thinking": thinking},
            ),
            raw=response,
        )

    @llm_chat_callback()
    @preserve_context_async
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        ollama_messages = self._convert_to_ollama_messages(messages)
        tools = kwargs.pop("tools", None)
        dynamic_context = self._get_dynamic_context_window(messages, tools)
        options = {**self._model_kwargs, "num_ctx": dynamic_context}
        think = kwargs.pop("think", None) or self.thinking
        format = kwargs.pop("format", "json" if self.json_mode else None)
        request = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "format": format,
            "tools": tools,
            "think": think,
            "options": options,
            "keep_alive": self.keep_alive,
        }

        async def gen() -> ChatResponseAsyncGen:
            response = await self.async_client.chat(**request)
            response_txt = ""
            thinking_txt = ""
            seen_tool_calls = set()
            all_tool_calls = []
            final_response = {}
            async for r in response:
                if r["message"]["content"] is None:
                    continue
                r = dict(r)
                response_txt += r["message"].get("content", "") or ""
                thinking_txt += r["message"].get("thinking", "") or ""
                new_tool_calls = [dict(t)
                                  for t in r["message"].get("tool_calls") or []]
                for tool_call in new_tool_calls:
                    tool_key = (str(tool_call["function"]["name"]), str(
                        tool_call["function"]["arguments"]))
                    if tool_key in seen_tool_calls:
                        continue
                    seen_tool_calls.add(tool_key)
                    all_tool_calls.append(tool_call)
                token_counts = self._get_response_token_counts(r)
                if token_counts:
                    r["usage"] = token_counts
                final_response = r
                # Safely set response_txt in final_response["message"]["content"]
                if "message" not in final_response:
                    final_response["message"] = {}
                final_response["message"]["content"] = response_txt
                yield ChatResponse(
                    message=ChatMessage(
                        content=response_txt,
                        role=r["message"].get("role", MessageRole.ASSISTANT),
                        additional_kwargs={
                            "tool_calls": all_tool_calls, "thinking": thinking_txt},
                    ),
                    delta=r["message"].get("content", ""),
                    raw=r,
                    additional_kwargs={
                        "thinking_delta": r["message"].get("thinking", None)},
                )
            if final_response:
                self._save_logs(request, final_response)
        return gen()

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return chat_to_completion_decorator(self.chat)(prompt, **kwargs)

    @llm_completion_callback()
    @preserve_context_async
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return await achat_to_completion_decorator(self.achat)(prompt, **kwargs)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        return stream_chat_to_completion_decorator(self.stream_chat)(prompt, **kwargs)

    @llm_completion_callback()
    @preserve_context_async
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        return await astream_chat_to_completion_decorator(self.astream_chat)(prompt, **kwargs)

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            llm_kwargs = llm_kwargs or {}
            llm_kwargs["format"] = output_cls.model_json_schema()
            messages = prompt.format_messages(**prompt_args)
            response = self.chat(messages, **llm_kwargs)
            return output_cls.model_validate_json(response.message.content or "")
        else:
            return super().structured_predict(output_cls, prompt, llm_kwargs, **prompt_args)

    @dispatcher.span
    @preserve_context_async
    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Model:
        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            llm_kwargs = llm_kwargs or {}
            llm_kwargs["format"] = output_cls.model_json_schema()
            messages = prompt.format_messages(**prompt_args)
            response = await self.achat(messages, **llm_kwargs)
            return output_cls.model_validate_json(response.message.content or "")
        else:
            return await super().astructured_predict(output_cls, prompt, llm_kwargs, **prompt_args)

    @dispatcher.span
    def stream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[Model, FlexibleModel], None, None]:
        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            def gen(
                output_cls: Type[Model],
                prompt: PromptTemplate,
                llm_kwargs: Dict[str, Any],
                prompt_args: Dict[str, Any],
            ) -> Generator[Union[Model, FlexibleModel], None, None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()
                messages = prompt.format_messages(**prompt_args)
                response_gen = self.stream_chat(messages, **llm_kwargs)
                cur_objects = None
                for response in response_gen:
                    try:
                        objects = process_streaming_objects(
                            response,
                            output_cls,
                            cur_objects=cur_objects,
                            allow_parallel_tool_calls=False,
                            flexible_mode=True,
                        )
                        cur_objects = objects if isinstance(
                            objects, list) else [objects]
                        yield objects
                    except Exception:
                        continue
            return gen(output_cls, prompt, llm_kwargs, prompt_args)
        else:
            return super().stream_structured_predict(output_cls, prompt, llm_kwargs, **prompt_args)

    @dispatcher.span
    @preserve_context_async
    async def astream_structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            async def gen(
                output_cls: Type[Model],
                prompt: PromptTemplate,
                llm_kwargs: Dict[str, Any],
                prompt_args: Dict[str, Any],
            ) -> AsyncGenerator[Union[Model, FlexibleModel], None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()
                messages = prompt.format_messages(**prompt_args)
                response_gen = await self.astream_chat(messages, **llm_kwargs)
                cur_objects = None
                async for response in response_gen:
                    try:
                        objects = process_streaming_objects(
                            response,
                            output_cls,
                            cur_objects=cur_objects,
                            allow_parallel_tool_calls=False,
                            flexible_mode=True,
                        )
                        cur_objects = objects if isinstance(
                            objects, list) else [objects]
                        yield objects
                    except Exception:
                        continue
            return gen(output_cls, prompt, llm_kwargs, prompt_args)
        else:
            return await super().astream_structured_predict(output_cls, prompt, llm_kwargs, **prompt_args)
