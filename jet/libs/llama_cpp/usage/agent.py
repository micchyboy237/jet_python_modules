from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    StreamCompletionResult,
    encode_image_to_base64,
    execute_tool_with_span,
    run_chat_stream,
)
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class Agent:
    """An extensible, stateful agent loop with built-in OpenTelemetry tracing, human-in-the-loop approvals, and retry mechanisms."""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = MODEL,
        max_turns: int = 5,
        system_prompt: str | None = None,
        require_approval: bool = False,
        approval_callback: Optional[Callable[[str, dict[str, Any]], bool]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_context_tokens: int = 32768,
        **llm_kwargs: Any,
    ):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.llm_kwargs = llm_kwargs
        self._tools_schema: list[dict[str, Any]] = []
        self._tool_registry: dict[str, Callable[..., Any]] = {}
        self.history: list[dict[str, Any]] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})
        self.tracer = trace.get_tracer(self.__class__.__name__)

        # Human-in-the-loop
        self.require_approval = require_approval
        self.approval_callback = approval_callback

        # Retry mechanism
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Context window management
        self.max_context_tokens = max_context_tokens

    def _count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a string (simplified)."""
        return len(text) // 4

    def _truncate_history(self) -> None:
        """Truncate history to fit within max_context_tokens."""
        if not self.history:
            return

        total_tokens = sum(
            self._count_tokens(str(msg.get("content", ""))) for msg in self.history
        )

        if total_tokens <= self.max_context_tokens:
            return

        truncated_history = []
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        non_system_msgs = [m for m in self.history if m.get("role") != "system"]

        for msg in reversed(non_system_msgs):
            msg_tokens = self._count_tokens(str(msg.get("content", "")))
            if total_tokens + msg_tokens > self.max_context_tokens:
                continue
            truncated_history.append(msg)
            total_tokens += msg_tokens

        truncated_history = list(reversed(truncated_history))
        self.history = system_msgs + truncated_history
        logger.info(f"🧹 Truncated history to {len(self.history)} messages.")

    def register_tool(
        self, schema: dict[str, Any], executor: Callable[..., Any]
    ) -> None:
        """Register a tool schema and its corresponding execution function."""
        name = schema.get("function", {}).get("name")
        if not name:
            raise ValueError(
                "Tool schema must include a 'function' object with a 'name'."
            )
        self._tools_schema.append(schema)
        self._tool_registry[name] = executor
        logger.debug(f"🔧 Registered tool: {name}")

    def on_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Hook executed when the LLM requests a tool.
        Override this in subclasses to add human-in-the-loop approvals,
        custom logging, or mock responses.
        """
        # Human-in-the-loop approval
        if self.require_approval:
            if self.approval_callback:
                approved = self.approval_callback(tool_name, arguments)
            else:
                user_input = input(
                    f"🛑 Approve tool call '{tool_name}' with arguments {json.dumps(arguments)}? (y/n): "
                )
                approved = user_input.lower() == "y"

            if not approved:
                logger.warning(f"❌ Tool call '{tool_name}' rejected by user/callback.")
                return {"error": f"Tool call '{tool_name}' was rejected."}

        # Retry mechanism
        executor = self._tool_registry.get(tool_name)
        if executor is None:
            logger.error(f"❌ Tool '{tool_name}' not found in registry!")
            return {"error": f"Tool '{tool_name}' is not available."}

        last_exception: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result = execute_tool_with_span(
                    tool_name, arguments, executor, strict=False
                )
                logger.info(
                    f"✅ Tool '{tool_name}' succeeded on attempt {attempt + 1}."
                )
                return result
            except Exception as e:
                last_exception = e
                delay = self.retry_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"⚠️ Tool '{tool_name}' failed on attempt {attempt + 1}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # All retries failed
        logger.error(f"❌ Tool '{tool_name}' failed after {self.max_retries} attempts.")
        return {
            "error": f"Tool '{tool_name}' failed after {self.max_retries} attempts: {last_exception}"
        }

    def clear_history(self) -> None:
        """Reset the conversation history, keeping the system prompt if it exists."""
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        self.history = system_msgs
        logger.info("🧹 Agent history cleared.")

    def run(
        self,
        prompt: str | None = None,
        image_source: str | None = None,
    ) -> StreamCompletionResult:
        """Execute the agent loop and return the final LLM response."""
        with self.tracer.start_as_current_span("agent_run") as span:
            span.set_attribute("agent.model", self.model)
            span.set_attribute("agent.max_turns", self.max_turns)
            span.set_attribute("agent.tools.count", len(self._tools_schema))

            # Truncate history before adding new messages
            self._truncate_history()

            if prompt or image_source:
                if image_source:
                    base64_img, mime_type = encode_image_to_base64(image_source)
                    content = [
                        {"type": "text", "text": prompt or ""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_img}"
                            },
                        },
                    ]
                    self.history.append({"role": "user", "content": content})
                    logger.info(f"🖼️ Added image to history: {image_source}")
                else:
                    self.history.append({"role": "user", "content": prompt})
                    logger.info(f"💬 Added prompt to history: {prompt[:50]}...")

            final_result = None
            for turn in range(1, self.max_turns + 1):
                logger.info(f"🔁 Agent Run: Turn {turn}/{self.max_turns}")
                result = run_chat_stream(
                    self.client,
                    messages=self.history,
                    model=self.model,
                    tools=self._tools_schema or None,
                    tool_choice="auto" if self._tools_schema else None,
                    **self.llm_kwargs,
                )
                final_result = result
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": result.content or "",
                }
                if result.has_tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.name,
                                "arguments": tc.raw_arguments,
                            },
                        }
                        for tc in result.tool_calls
                    ]
                self.history.append(assistant_msg)
                if not result.has_tool_calls:
                    logger.info("✅ Agent loop complete. No more tool calls.")
                    break
                for tc in result.tool_calls:
                    tool_result = self.on_tool_call(tc.name, tc.arguments)
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result, default=str),
                        }
                    )
            else:
                logger.warning(f"⚠️ Agent run hit max_turns limit ({self.max_turns}).")

            span.set_status(Status(StatusCode.OK))
            return final_result
