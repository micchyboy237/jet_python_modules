# jet_python_modules/jet/libs/llama_cpp/usage/agent.py
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
from jet.libs.llama_cpp.usage.context_window import ContextWindow
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)


class Agent:
    """An extensible, stateful agent loop with built-in OpenTelemetry tracing,
    human-in-the-loop approvals, retry mechanisms, and encapsulated context management."""

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
        max_context_tokens: int = 16384,
        **llm_kwargs: Any,
    ):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.llm_kwargs = llm_kwargs
        self._tools_schema: list[dict[str, Any]] = []
        self._tool_registry: dict[str, Callable[..., Any]] = {}
        self.tracer = trace.get_tracer(self.__class__.__name__)
        self.require_approval = require_approval
        self.approval_callback = approval_callback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_context_tokens = max_context_tokens

        # Extract base_url from client to ensure token counting uses the same server
        base_url = str(client.base_url) if hasattr(client, "base_url") else None

        self._context = ContextWindow(
            max_tokens=max_context_tokens,
            model=model,
            base_url=base_url,
        )
        if system_prompt:
            self._context.append({"role": "system", "content": system_prompt})

    @property
    def history(self) -> list[dict[str, Any]]:
        """Backward-compatible read access to context history."""
        return self._context.get_messages()

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
        if self.require_approval:
            if self.approval_callback:
                approved = self.approval_callback(tool_name, arguments)
            else:
                user_input = input(
                    f"🛑 Approve tool call '{tool_name}' with arguments "
                    f"{json.dumps(arguments)}? (y/n): "
                )
                approved = user_input.lower() == "y"

            if not approved:
                logger.warning(f"❌ Tool call '{tool_name}' rejected by user/callback.")
                return {"error": f"Tool call '{tool_name}' was rejected."}

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
                delay = self.retry_delay * (2**attempt)
                logger.warning(
                    f"⚠️ Tool '{tool_name}' failed on attempt {attempt + 1}: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        logger.error(f"❌ Tool '{tool_name}' failed after {self.max_retries} attempts.")
        return {
            "error": (
                f"Tool '{tool_name}' failed after {self.max_retries} attempts: "
                f"{last_exception}"
            )
        }

    def clear_history(self) -> None:
        """Reset the conversation history, keeping the system prompt if it exists."""
        self._context.clear(preserve_system=True)

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
            span.set_attribute("agent.context.tokens", self._context.total_tokens())
            span.set_attribute("agent.context.messages", self._context.message_count)

            self._context.truncate_if_needed()

            if prompt or image_source:
                if image_source:
                    base64_img, mime_type = encode_image_to_base64(image_source)
                    self._context.append_image(prompt, base64_img, mime_type)
                else:
                    self._context.append({"role": "user", "content": prompt})
                    logger.info(f"💬 Added prompt to context: {prompt[:50]}...")

            final_result = None
            for turn in range(1, self.max_turns + 1):
                logger.info(f"🔁 Agent Run: Turn {turn}/{self.max_turns}")

                result = run_chat_stream(
                    self.client,
                    messages=self._context.get_messages(),
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

                self._context.append(assistant_msg)

                if not result.has_tool_calls:
                    logger.info("✅ Agent loop complete. No more tool calls.")
                    break

                for tc in result.tool_calls:
                    tool_result = self.on_tool_call(tc.name, tc.arguments)
                    self._context.append(
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
