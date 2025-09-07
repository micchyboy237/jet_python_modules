from jet.llm.mlx.mlx_utils import parse_tool_calls
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from typing import List, Literal, Optional, Union, Iterator, AsyncIterator, Any, Type, TypeVar
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import ChatTemplateArgs, RoleMapping, Tool
from jet.models.model_types import LLMModelType
from uuid import uuid4
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages.tool import ToolCall
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)


class ChatMLX(BaseChatModel):
    """Chat model integration for MLX framework using LangChain interface."""

    # Explicitly declare fields with defaults or mark as optional
    mlx_client: Optional[MLX] = Field(
        default=None, description="MLX client instance")
    model: LLMModelType = Field(
        default="qwen3-1.7b-4bit", description="Model type")
    temperature: Optional[float] = Field(
        default=0.8, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        default=128, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(
        default=0.9, description="Top-p sampling probability")
    top_k: Optional[int] = Field(
        default=40, description="Top-k sampling value")

    class Config:
        """Pydantic configuration to allow arbitrary types and dynamic attributes."""
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
        extra = "allow"  # Allow extra attributes to be set dynamically

    @property
    def provider(self) -> str:
        """Return the provider of the chat model."""
        return "mlx"

    @property
    def name(self) -> str:
        """Return the name of the chat model."""
        return self.model

    def __init__(
        self,
        model: LLMModelType = "qwen3-1.7b-4bit",
        adapter_path: Optional[str] = None,
        draft_model: Optional[LLMModelType] = None,
        trust_remote_code: bool = False,
        chat_template: Optional[str] = None,
        use_default_chat_template: bool = True,
        chat_template_args: Optional[ChatTemplateArgs] = None,
        dbname: str = "mlx_chat_history_db1",
        user: str = "default_user",
        password: str = "default_password",
        host: str = "localhost",
        port: int = 5432,
        overwrite_db: bool = False,
        session_id: Optional[str] = None,
        with_history: bool = False,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        device: Optional[Literal["cpu", "mps"]] = "mps",
        prompt_cache: Optional[List[Any]] = None,
        temperature: Optional[float] = 0.8,
        max_tokens: Optional[int] = 128,
        top_p: Optional[float] = 0.9,
        top_k: Optional[int] = 40,
        **kwargs: Any
    ):
        """Initialize ChatMLX with MLX client and configuration."""
        # Provide a fallback chat template for models like llama-3.2-3b-instruct
        if chat_template is None and use_default_chat_template:
            chat_template = (
                "{% for message in messages %}"
                "{{ '<|startoftext|>' if loop.first else '' }}"
                "{{ '<|user|> ' if message.role == 'user' else '<|assistant|> ' }}"
                "{{ message.content }}"
                "{{ '<|endoftext|>' if loop.last else '' }}"
                "{% endfor %}"
            )

        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs
        )
        try:
            self.mlx_client = MLXModelRegistry.load_model(
                model=model,
                adapter_path=adapter_path,
                draft_model=draft_model,
                trust_remote_code=trust_remote_code,
                chat_template=chat_template,
                use_default_chat_template=use_default_chat_template,
                chat_template_args=chat_template_args,
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                overwrite_db=overwrite_db,
                session_id=session_id,
                with_history=with_history,
                seed=seed,
                log_dir=log_dir,
                device=device,
                prompt_cache=prompt_cache,
                verbose=True
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize MLX client: {str(e)}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k

    def _convert_messages_to_mlx_format(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to MLX message format."""
        if not messages:
            raise ValueError("Messages list cannot be empty or None")
        mlx_messages = []
        for message in messages:
            role: str
            content: str = message.content if isinstance(
                message.content, str) else ""

            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ToolMessage):
                role = "tool"
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

            mlx_messages.append({
                "role": role,
                "content": content,
                "tool_call_id": message.tool_call_id if isinstance(message, ToolMessage) else None,
                "tool_calls": message.tool_calls if isinstance(message, AIMessage) and message.tool_calls else None
            })
        return mlx_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Generate a chat response using MLX client."""
        mlx_messages = self._convert_messages_to_mlx_format(messages)

        tools: List[Tool] = kwargs.pop("tools", None)
        if "functions" in kwargs:
            functions = kwargs.pop("functions")
            tools = [Tool(type="function", function=fn) for fn in functions]

        response = self.mlx_client.chat(
            messages=mlx_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=stop,
            tools=tools,
            **kwargs
        )

        content = response["choices"][0]["message"]["content"] if response.get(
            "choices") else ""

        tool_calls = parse_tool_calls(response["content"])
        formatted_tool_calls: List[ToolCall] = [
            {"name": tc["name"], "args": tc["arguments"],
                "id": None, "type": "tool_call"}
            for tc in tool_calls]

        chat_generation = ChatGeneration(
            message=AIMessage(
                content=content,
                tool_calls=formatted_tool_calls,
                id=str(uuid4())
            ),
            generation_info=response
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat responses using MLX client."""
        mlx_messages = self._convert_messages_to_mlx_format(messages)

        tools: List[Tool] = kwargs.pop("tools", None)
        if "functions" in kwargs:
            functions = kwargs.pop("functions")
            tools = [Tool(type="function", function=fn) for fn in functions]

        for response in self.mlx_client.stream_chat(
            messages=mlx_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=stop,
            tools=tools,
            **kwargs
        ):
            content = response["choices"][0]["message"]["content"] if response.get(
                "choices") else ""

            tool_calls = []
            if response["choices"][0]["finish_reason"]:
                tool_calls = parse_tool_calls(response["content"])
            formatted_tool_calls: List[ToolCall] = [
                {"name": tc["name"], "args": tc["arguments"],
                    "id": None, "type": "tool_call"}
                for tc in tool_calls]

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content,
                    tool_calls=formatted_tool_calls,
                    id=str(uuid4())
                ),
                generation_info=response
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Asynchronously generate a chat response using MLX client."""
        # MLX client doesn't support async natively, so we use sync version
        return self._generate(messages, stop, run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chat responses using MLX client."""
        # MLX client doesn't support async streaming natively, so we use sync streaming
        for chunk in self._stream(messages, stop, None, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text)
            yield chunk

    async def ainvoke(
        self,
        messages: List[BaseMessage],
        output_format: Optional[Type[T]] = None
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        """Asynchronously invoke the chat model with messages."""
        result = await self._agenerate(messages, stop=None, run_manager=None)
        completion = result.generations[0].message.content if result.generations else ""
        if output_format and issubclass(output_format, BaseModel):
            try:
                parsed = output_format.parse_raw(completion)
                return ChatInvokeCompletion(completion=parsed)
            except ValidationError as e:
                raise ValueError(f"Failed to parse completion as {output_format.__name__}: {str(e)}")
        return ChatInvokeCompletion(completion=completion)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-mlx"
