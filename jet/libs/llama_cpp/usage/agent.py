from __future__ import annotations

import json
import logging
from typing import Any, Callable

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
    """An extensible, stateful agent loop with built-in OpenTelemetry tracing.

    Encapsulates the LLM, tool registry, and conversation history, providing
    a unified trace for the entire multi-turn execution in Phoenix.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str = MODEL,
        max_turns: int = 5,
        system_prompt: str | None = None,
        **llm_kwargs: Any,
    ):
        self.client = client
        self.model = model
        self.max_turns = max_turns
        self.llm_kwargs = llm_kwargs

        self._tools_schema: list[dict[str, Any]] = []
        self._tool_registry: dict[str, Callable[..., Any]] = {}

        # Initialize conversation history
        self.history: list[dict[str, Any]] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

        self.tracer = trace.get_tracer(self.__class__.__name__)

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
        executor = self._tool_registry.get(tool_name)
        if executor is None:
            logger.error(f"❌ Tool '{tool_name}' not found in registry!")
            return {"error": f"Tool '{tool_name}' is not available."}

        return execute_tool_with_span(tool_name, arguments, executor, strict=False)

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

            # 1. Append new user input to history
            if prompt or image_source:
                if image_source:
                    # Handle vision encoding internally so history remains accurate
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

            # 2. Agent Loop
            for turn in range(1, self.max_turns + 1):
                logger.info(f"🔁 Agent Run: Turn {turn}/{self.max_turns}")

                # Call LLM with current history
                result = run_chat_stream(
                    self.client,
                    messages=self.history,
                    model=self.model,
                    tools=self._tools_schema or None,
                    tool_choice="auto" if self._tools_schema else None,
                    **self.llm_kwargs,
                )

                final_result = result

                # 3. Append assistant response to history
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

                # 4. Check if done
                if not result.has_tool_calls:
                    logger.info("✅ Agent loop complete. No more tool calls.")
                    break

                # 5. Execute tools and append results to history
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
