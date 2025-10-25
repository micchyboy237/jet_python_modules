from __future__ import annotations
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from jet.adapters.llama_cpp.llm import ChatMessage, LlamacppLLM
from operator import itemgetter
from typing import (
    Callable,
    Dict,
    Type,
    Union,
    cast,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    ChatMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from jet.logger import logger
from jet.transformers.formatters import format_json

class ChatLlamaCpp(BaseChatModel):
    """
    LangChain chat model wrapper for LlamacppLLM.
    Extends BaseChatModel and mirrors the structure of ChatOllama for consistency.
    """
    model_config = {"arbitrary_types_allowed": True}
    model: str = Field(..., description="Model identifier (e.g., 'qwen3-instruct-2507:4b')")
    base_url: str = Field(
        default="http://shawn-pc.local:8080/v1",
        description="Base URL of the llama.cpp OpenAI-compatible server",
    )
    api_key: str = Field(default="sk-1234", description="API key (required by OpenAI spec)")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts on failure")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    llm: LlamacppLLM = Field(default=None, exclude=True, repr=False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.llm = LlamacppLLM(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            max_retries=self.max_retries,
            verbose=self.verbose,
        )

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-chat"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[ChatMessage]:
        """Convert LangChain BaseMessage list to llama.cpp ChatMessage format."""
        result: List[ChatMessage] = []
        for msg in messages:
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
                "function": "tool",
            }
            role = role_map.get(msg.type, "user")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            item: ChatMessage = {"role": role, "content": content}
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                item["tool_call_id"] = msg.tool_call_id
            result.append(item)
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_messages = self._convert_messages(messages)
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if stop:
            params["stop"] = stop
        if tools:
            available_functions: Dict[str, Callable] = {}
            for tool in tools:
                func_name = tool["function"]["name"]
                # Use the actual tool callable from LangChain's BaseTool
                available_functions[func_name] = lambda **args: BaseTool(**tool).run(args)
            response = self.llm.chat_with_tools(
                messages=llm_messages,
                tools=tools,
                available_functions=available_functions,
                **params,
            )
        else:
            response = self.llm.chat(
                messages=llm_messages,
                stream=False,
                **params,
            )
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_messages = self._convert_messages(messages)
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if stop:
            params["stop"] = stop
        if tools:
            available_functions: Dict[str, Callable] = {}
            for tool in tools:
                func_name = tool["function"]["name"]
                # Use the actual tool callable from LangChain's BaseTool
                available_functions[func_name] = lambda **args: BaseTool(**tool).run(args)
            response = await self.llm.achat_with_tools(
                messages=llm_messages,
                tools=tools,
                available_functions=available_functions,
                **params,
            )
        else:
            response = await self.llm.achat(
                messages=llm_messages,
                stream=False,
                **params,
            )
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        llm_messages = self._convert_messages(messages)
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if stop:
            params["stop"] = stop
        stream = self.llm.chat_stream(messages=llm_messages, **params)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            content = delta.content if delta and delta.content is not None else ""
            if content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        llm_messages = self._convert_messages(messages)
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if stop:
            params["stop"] = stop
        async for chunk in self.llm.achat_stream(messages=llm_messages, **params):
            delta = chunk.choices[0].delta if chunk.choices else None
            content = delta.content if delta and delta.content is not None else ""
            if content:
                yield ChatGenerationChunk(message=AIMessageChunk(content=content))

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, bool, str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        logger.info(f"Binding tools: {format_json(formatted_tools)}")
        tool_names = [ft["function"]["name"] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice["function"]["name"] == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                chosen = [
                    f for f in formatted_tools if f["function"]["name"] == tool_choice
                ]
                if not chosen:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like 
                    this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )
        kwargs["tool_choice"] = tool_choice
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema."""
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and is_basemodel_subclass(schema)
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        tool_choice = {"type": "function", "function": {"name": tool_name}}
        llm = self.bind_tools([schema], tool_choice=tool_choice)
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast(Type, schema)], first_tool_only=True
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

class CallableWithToolSpec:
    """Wrapper to preserve OpenAI tool spec on callable objects (e.g. lambdas)."""
    def __init__(self, func: Callable, tool_spec: Dict[str, Any]):
        self.func = func
        self.tool_spec = tool_spec
        self.__name__ = tool_spec["function"]["name"]
        self.__doc__ = tool_spec["function"].get("description", "No description available.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)
