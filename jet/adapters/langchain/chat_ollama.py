from pydantic import BaseModel
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Sequence, Union
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama as BaseChatOllama
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from ollama import Options
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.logger import logger
from jet.transformers.formatters import format_json

DETERMINISTIC_LLM_SETTINGS = {
    "seed": 42,
    "temperature": 0,
    "num_keep": 0,
    "num_predict": -1,
}


class ChatOllama(BaseChatOllama):
    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        options = {**DETERMINISTIC_LLM_SETTINGS, **(kwargs.pop("options", {}))}
        super().__init__(model=model, base_url=base_url, **options, **kwargs)
        self._chat_logger = ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method="chat")

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        if self.stop is not None and stop is not None:
            msg = "`stop` found in both the input and default params."
            raise ValueError(msg)
        if self.stop is not None:
            stop = self.stop
        options_dict = {
            "mirostat": self.mirostat,
            "mirostat_eta": self.mirostat_eta,
            "mirostat_tau": self.mirostat_tau,
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_thread": self.num_thread,
            "num_predict": self.num_predict,
            "repeat_last_n": self.repeat_last_n,
            "repeat_penalty": self.repeat_penalty,
            "temperature": kwargs.pop("temperature", self.temperature),
            "seed": self.seed,
            "stop": self.stop if stop is None else stop,
            "tfs_z": self.tfs_z,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        params = {
            "messages": ollama_messages,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "think": kwargs.pop("reasoning", self.reasoning),
            "format": kwargs.pop("format", self.format),
            "options": Options(**options_dict),
            "keep_alive": kwargs.pop("keep_alive", self.keep_alive),
            **kwargs,
        }
        if tools := kwargs.get("tools"):
            params["tools"] = tools
        return params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        logger.gray("Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.dict() for msg in messages],
            "stop": stop,
            "kwargs": kwargs,
        }))
        result = super()._generate(messages, stop, run_manager, **kwargs)
        for generation in result.generations:
            logger.teal(generation.message.content, flush=True)
            self._chat_logger.log_interaction(
                messages=messages,
                response=generation.message.dict(),
                model=self.model,
                tools=kwargs.get("tools"),
            )
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        logger.gray("Stream Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.dict() for msg in messages],
            "stop": stop,
            "kwargs": kwargs,
        }))
        for chunk in super()._stream(messages, stop, run_manager, **kwargs):
            content = chunk.message.content if isinstance(
                chunk.message.content, str) else str(chunk.message.content)
            logger.teal(content, flush=True)
            self._chat_logger.log_interaction(
                messages=messages,
                response=chunk.message.dict(),
                model=self.model,
                tools=kwargs.get("tools"),
            )
            yield chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        logger.gray("Async Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.dict() for msg in messages],
            "stop": stop,
            "kwargs": kwargs,
        }))
        result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        for generation in result.generations:
            logger.teal(generation.message.content, flush=True)
            self._chat_logger.log_interaction(
                messages=messages,
                response=generation.message.dict(),
                model=self.model,
                tools=kwargs.get("tools"),
            )
        return result

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        logger.gray("Async Stream Chat LLM Settings:")
        logger.info(format_json({
            "messages": [msg.dict() for msg in messages],
            "stop": stop,
            "kwargs": kwargs,
        }))
        async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
            content = chunk.message.content if isinstance(
                chunk.message.content, str) else str(chunk.message.content)
            logger.teal(content, flush=True)
            self._chat_logger.log_interaction(
                messages=messages,
                response=chunk.message.dict(),
                model=self.model,
                tools=kwargs.get("tools"),
            )
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        logger.info(
            f"Binding tools: {format_json([tool if isinstance(tool, dict) else str(tool) for tool in tools])}")
        return super().bind_tools(tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[dict, type],
        *,
        method: str = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        logger.info(
            f"Configuring structured output with method: {method}, schema: {format_json(schema if isinstance(schema, dict) else str(schema))}")
        return super().with_structured_output(schema, method=method, include_raw=include_raw, **kwargs)
