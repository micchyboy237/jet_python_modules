import os
from typing import List, Any, Optional, Iterator, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun

from jet.llm.config import DEFAULT_LOG_DIR
from jet.llm.logger_utils import ChatLogger
from jet.logger import CustomLogger
from jet.logger.config import DEFAULT_LOGGER
from jet.transformers.formatters import format_json
from jet.utils.text import format_sub_dir


class ChatLlamaCpp(ChatOpenAI):
    def __init__(
        self,
        *args,
        model: str = "qwen3-instruct-2507:4b",
        temperature: float = 0.0,
        base_url: str = "http://shawn-pc.local:8080/v1",
        verbosity: str = "high",
        verbose: bool = True,
        agent_name: Optional[str] = None,
        log_dir: str = DEFAULT_LOG_DIR,
        logger: Optional[CustomLogger] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            model=model,
            temperature=temperature,
            base_url=base_url,
            **kwargs,
        )

        if agent_name and (not logger or not logger.log_file):
            log_dir = os.path.join(log_dir, format_sub_dir(agent_name))

        self._model: str = model
        self._agent_name: Optional[str] = agent_name
        self._log_dir: str = log_dir
        self._verbose: bool = verbose

        self._logger = logger or CustomLogger(DEFAULT_LOGGER, filename=f"{log_dir}/main.log")
        self._chat_logger: Optional[ChatLogger] = (
            ChatLogger(log_dir=self._log_dir) if self._verbose else None
        )

        # Log each init argument
        self._log("Initialized ChatLlamaCpp:\n%s", format_json({
            "model": model,
            "temperature": temperature,
            "agent_name": agent_name,
        }))
        if kwargs:
            self._log("additional kwargs: %s", kwargs)

    # --------------------------------------------------------------------- #
    # Helper
    # --------------------------------------------------------------------- #
    def _log(self, message: str, *args: Any) -> None:
        """Log only when verbose is enabled."""
        if self._verbose:
            self._logger.info(message, *args)

    # --------------------------------------------------------------------- #
    # Sync generation (invoke)
    # --------------------------------------------------------------------- #
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate full response using streaming; log exactly once."""
        self._logger.info("Starting _generate")
        self._logger.gray(f"\nMessages ({len(messages)}):")
        self._logger.debug(format_json(messages))
        
        if kwargs.get("tools"):
            self._logger.gray("\nTools:")
            self._logger.debug(format_json(kwargs["tools"]))

        text_content = ""
        chunks: List[ChatGenerationChunk] = []
        tool_call_chunks: List[dict] = []
        reasoning_chunks: List[str] = []
        invalid_tool_calls: List[dict] = []

        for chunk in self._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks.append(chunk)
            msg = chunk.message

            # Only accumulate from legacy .content if content_blocks are absent
            if not (hasattr(msg, "content_blocks") and msg.content_blocks):
                if msg.content:
                    text_content += msg.content
            else:
                # Process structured content blocks
                for block in msg.content_blocks:
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        text_content += block["text"]
                    elif btype == "tool_call_chunk":
                        tool_call_chunks.append(block)
                        text_content += block.get("args", "")
                    elif btype == "reasoning" and block.get("reasoning"):
                        reasoning_chunks.append(block["reasoning"])
                    elif btype == "invalid_tool_call":
                        invalid_tool_calls.append(block)

        if not chunks:
            self._log("No chunks from _stream, falling back to super()._generate")
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        # === Fix: Propagate first tool_call_chunk's id/name to all others ===
        if tool_call_chunks:
            first = tool_call_chunks[0]
            first_id = first.get("id")
            first_name = first.get("name")
            for chunk_idx, chunk in enumerate(tool_call_chunks):
                chunk["index"] = chunk_idx
                if first_id and not chunk.get("id"):
                    chunk["id"] = first_id
                if first_name and not chunk.get("name"):
                    chunk["name"] = first_name

        # Build final message: content is always str (empty if no text)
        final_message_kwargs = {
            "content": "" if tool_call_chunks else text_content
        }

        if tool_call_chunks:
            final_message_kwargs["tool_call_chunks"] = [{
                **tool_call_chunks[0],
                "args": text_content
            }]

        if invalid_tool_calls:
            final_message_kwargs["invalid_tool_calls"] = invalid_tool_calls

        if reasoning_chunks:
            final_message_kwargs["additional_kwargs"] = {
                "reasoning": "\n".join(reasoning_chunks)
            }

        final_message = AIMessageChunk(**final_message_kwargs)

        result = ChatResult(
            generations=[ChatGenerationChunk(message=final_message)]
        )

        # ---- single log_interaction call -------------------------------- #
        if self._verbose and self._chat_logger is not None:
            self._log("Logging interaction (sync generate)")
            invocation_params = self._get_invocation_params(stop=stop, **kwargs)
            metadata = {
                "model": self._model,
                "agent_name": self._agent_name,
                "method": "chat",
                "invocation": invocation_params,
            }
            self._chat_logger.log_interaction(
                messages=messages,
                response=text_content,
                **metadata,
            )

        return result

    # --------------------------------------------------------------------- #
    # Async generation (ainvoke)
    # --------------------------------------------------------------------- #
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate using streaming; log exactly once."""
        self._logger.info("Starting _agenerate")
        self._logger.gray(f"\nMessages ({len(messages)}):")
        self._logger.debug(format_json(messages))
        
        if kwargs.get("tools"):
            self._logger.gray("\nTools:")
            self._logger.debug(format_json(kwargs["tools"]))

        text_content = ""
        chunks: List[ChatGenerationChunk] = []
        tool_call_chunks: List[dict] = []
        reasoning_chunks: List[str] = []
        invalid_tool_calls: List[dict] = []

        async for chunk in self._astream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks.append(chunk)
            msg = chunk.message

            # Only accumulate from legacy .content if content_blocks are absent
            if not (hasattr(msg, "content_blocks") and msg.content_blocks):
                if msg.content:
                    text_content += msg.content
            else:
                # Process structured content blocks
                for block in msg.content_blocks:
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        text_content += block["text"]
                    elif btype == "tool_call_chunk":
                        tool_call_chunks.append(block)
                        text_content += block.get("args", "")
                    elif btype == "reasoning" and block.get("reasoning"):
                        reasoning_chunks.append(block["reasoning"])
                    elif btype == "invalid_tool_call":
                        invalid_tool_calls.append(block)

        if not chunks:
            self._log("No chunks from _astream, falling back to super()._agenerate")
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

        # === Fix: Propagate first tool_call_chunk's id/name to all others ===
        if tool_call_chunks:
            first = tool_call_chunks[0]
            first_id = first.get("id")
            first_name = first.get("name")
            for chunk_idx, chunk in enumerate(tool_call_chunks):
                chunk["index"] = chunk_idx
                if first_id and not chunk.get("id"):
                    chunk["id"] = first_id
                if first_name and not chunk.get("name"):
                    chunk["name"] = first_name

        # Build final message: content is always str (empty if no text)
        final_message_kwargs = {
            "content": "" if tool_call_chunks else text_content
        }

        if tool_call_chunks:
            final_message_kwargs["tool_call_chunks"] = [{
                **tool_call_chunks[0],
                "args": text_content
            }]

        if invalid_tool_calls:
            final_message_kwargs["invalid_tool_calls"] = invalid_tool_calls

        if reasoning_chunks:
            final_message_kwargs["additional_kwargs"] = {
                "reasoning": "\n".join(reasoning_chunks)
            }

        final_message = AIMessageChunk(**final_message_kwargs)

        result = ChatResult(
            generations=[ChatGenerationChunk(message=final_message)]
        )

        # ---- single log_interaction call -------------------------------- #
        if self._verbose and self._chat_logger is not None:
            self._log("Logging interaction (async generate)")
            invocation_params = self._get_invocation_params(stop=stop, **kwargs)
            metadata = {
                "model": self._model,
                "agent_name": self._agent_name,
                "method": "achat",
                "invocation": invocation_params,
            }
            if run_manager and hasattr(run_manager, "get_sync") and callable(run_manager.get_sync):
                run_manager.get_sync()(
                    self._chat_logger.log_interaction,
                    messages,
                    response=text_content,
                    **metadata,
                )
            else:
                self._chat_logger.log_interaction(
                    messages=messages,
                    response=text_content,
                    **metadata,
                )

        return result

    # --------------------------------------------------------------------- #
    # Sync streaming (stream)
    # --------------------------------------------------------------------- #
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Yield chunks and print them; **no** log_interaction."""
        self._log("Starting _stream for %s messages", len(messages))

        for chunk in super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if self._verbose:
                content_blocks = getattr(chunk.message, "content_blocks", None)
                if content_blocks:
                    for block in content_blocks:
                        block_type = block.get("type")
                        if block_type == "tool_call_chunk":
                            args = block.get("args", "")
                            self._logger.teal(args, flush=True)
                        elif block_type == "text":
                            text = block.get("text", "")
                            self._logger.teal(text, flush=True)
                        elif block_type == "reasoning":
                            reasoning = block.get("reasoning", "")
                            self._logger.teal(f"[Reasoning] {reasoning}", flush=True)
                        elif block_type == "invalid_tool_call":
                            name = block.get("name", "unknown")
                            error = block.get("error", "unknown error")
                            self._logger.teal(f"[Invalid Tool Call] {name}: {error}", flush=True)
                elif chunk.message.content:
                    self._logger.teal(chunk.message.content, flush=True)
            yield chunk

        self._log("Finished _stream")

    # --------------------------------------------------------------------- #
    # Async streaming (astream)
    # --------------------------------------------------------------------- #
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Yield async chunks and print them; **no** log_interaction."""
        self._log("Starting _astream for %s messages", len(messages))

        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            if self._verbose:
                content_blocks = getattr(chunk.message, "content_blocks", None)
                if content_blocks:
                    for block in content_blocks:
                        block_type = block.get("type")
                        if block_type == "tool_call_chunk":
                            args = block.get("args", "")
                            self._logger.teal(args, flush=True)
                        elif block_type == "text":
                            text = block.get("text", "")
                            self._logger.teal(text, flush=True)
                        elif block_type == "reasoning":
                            reasoning = block.get("reasoning", "")
                            self._logger.teal(f"[Reasoning] {reasoning}", flush=True)
                        elif block_type == "invalid_tool_call":
                            name = block.get("name", "unknown")
                            error = block.get("error", "unknown error")
                            self._logger.teal(f"[Invalid Tool Call] {name}: {error}", flush=True)
                elif chunk.message.content:
                    self._logger.teal(chunk.message.content, flush=True)
            yield chunk

        self._log("Finished _astream")