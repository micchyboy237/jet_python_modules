# jet_python_modules/jet/adapters/langchain/chat_llama_cpp.py
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from jet.adapters.llama_cpp.llm import ChatMessage, LlamacppLLM
from langchain_core.runnables import RunnableLambda

from jet.logger import logger
from jet.transformers.formatters import format_json


class ChatLlamaCpp(BaseChatModel):
    """
    LangChain chat model wrapper for LlamacppLLM.
    Extends BaseChatModel and mirrors the structure of ChatOllama for consistency.
    """
    model_config = {"arbitrary_types_allowed": True}  # Optional: for future-proofing

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
            item: ChatMessage = {"role": role, "content": content}  # type: ignore[typedd]
            result.append(item)
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
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
        response = self.llm.chat(
            messages=llm_messages,
            stream=False,
            **params,
        )
        # FIX: Use proper AIMessage instead of raw dict
        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
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
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        """Bind tools to the chat model using OpenAI-compatible tool calling."""
        from langchain_core.utils.function_calling import convert_to_openai_tool
        formatted_tools = [
            convert_to_openai_tool(tool) if not isinstance(tool, dict) else tool
            for tool in tools
        ]
        available_functions = {}
        for tool in tools:
            if hasattr(tool, "name") and callable(tool):
                available_functions[tool.name] = tool
            elif isinstance(tool, dict) and "function" in tool:
                name = tool["function"]["name"]
                if hasattr(tool, "invoke"):
                    available_functions[name] = lambda args, t=tool: t.invoke(args)
                else:
                    pass
        logger.info(f"Binding {len(formatted_tools)} tools: {[t['function']['name'] for t in formatted_tools]}")
        
        def _tool_caller(messages: List[BaseMessage], **runtime_kwargs) -> AIMessage:
            llm_messages = self._convert_messages(messages)
            call_kwargs = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **runtime_kwargs,
                **kwargs,
            }
            response = self.llm.chat_with_tools(
                messages=llm_messages,
                tools=formatted_tools,
                available_functions=available_functions,
                **call_kwargs
            )
            return AIMessage(
                content=response,
                tool_calls=getattr(response, "tool_calls", [])
            )
        
        bound = RunnableLambda(_tool_caller).with_config(run_name="ChatLlamaCppWithTools")
        return bound

    def with_structured_output(
        self,
        schema: Any,
        **kwargs: Any,
    ) -> Runnable:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema()
        else:
            json_schema = schema
        logger.info(
            f"Configuring structured output with options: {format_json(kwargs)}\n\nSchema: {format_json(json_schema)}")

        def _invoke(messages: List[BaseMessage]) -> BaseModel:
            llm_messages = self._convert_messages(messages)
            return self.llm.chat_structured(llm_messages, schema, **kwargs)

        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(_invoke)
