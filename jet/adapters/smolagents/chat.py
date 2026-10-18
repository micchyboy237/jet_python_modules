"""SmolAgents LLM adapter for local llama.cpp servers using jet observability."""

from __future__ import annotations

import json
from typing import Any

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.llm_utils_observed import chat as jet_chat
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from smolagents.models import ChatMessage, MessageRole, Model, TokenUsage
from smolagents.tools import Tool


class ChatLlamaCpp(Model):
    """A SmolAgents compatible model backed by jet's observed llama.cpp utilities.

    This model integrates local llama.cpp inference into the SmolAgents framework
    by leveraging jet's built-in observability, tool handling, and structured output features.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        base_url: str | None = LLM_BASE_URL,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float = 1.1,
        enable_thinking: bool = False,
        project_name: str = "smolagents-agent",
        **kwargs: Any,
    ):
        """Initialize the ChatLlamaCpp model.

        Args:
            model: The model identifier to use.
            base_url: Base URL for the llama.cpp server. Defaults to LLM_BASE_URL.
            temperature: Sampling temperature.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top-p sampling parameter.
            top_k: Top-k sampling parameter.
            repeat_penalty: Repeat penalty.
            enable_thinking: Enable thinking/reasoning tokens if supported.
            project_name: Phoenix project name for observability traces.
            **kwargs: Additional keyword arguments to forward to jet_chat.
        """
        super().__init__(model_id=model, **kwargs)
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.enable_thinking = enable_thinking
        self.project_name = project_name
        self.extra_kwargs = kwargs

    def _convert_messages_to_openai_format(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, Any]]:
        """Convert SmolAgents ChatMessages to OpenAI-compatible message format."""
        openai_messages = []
        for msg in messages:
            role = msg.role.value
            # Map SmolAgents roles to OpenAI roles if necessary
            if role == "tool-call":
                role = "assistant"
            elif role == "tool-response":
                role = "user"

            content = msg.content
            if isinstance(content, list):
                # Handle multi-modal content if present
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)

            openai_messages.append({"role": role, "content": content})

        return openai_messages

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Generate a response using the local llama.cpp model.

        Args:
            messages: List of chat messages.
            stop_sequences: Optional stop sequences.
            response_format: Optional response format for structured output.
            tools_to_call_from: Optional list of tools available for the model.
            **kwargs: Additional keyword arguments.

        Returns:
            A ChatMessage containing the model's response.
        """
        # Convert SmolAgents messages to OpenAI format
        openai_messages = self._convert_messages_to_openai_format(messages)

        # Prepare tools for jet_chat if provided
        tools = None
        tool_registry = None
        if tools_to_call_from:
            # Convert SmolAgents tools to OpenAI format
            tools = []
            tool_registry = {}
            for tool in tools_to_call_from:
                # Assuming tool has name, description, and inputs/parameters
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool.inputs,
                            "required": [
                                k
                                for k, v in tool.inputs.items()
                                if not v.get("nullable", False)
                            ],
                        },
                    },
                }
                tools.append(tool_def)
                tool_registry[tool.name] = tool.forward

        # Call jet_chat with observability
        result: StreamCompletionResult = jet_chat(
            prompt_or_messages=openai_messages,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            enable_thinking=self.enable_thinking,
            stop=stop_sequences,
            tools=tools,
            tool_registry=tool_registry,
            response_format=response_format,
            project_name=self.project_name,
            **self.extra_kwargs,
            **kwargs,
        )

        # Parse tool calls if present
        tool_calls = None
        if result.tool_calls:
            from smolagents.models import (
                ChatMessageToolCall,
                ChatMessageToolCallFunction,
            )

            tool_calls = [
                ChatMessageToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ChatMessageToolCallFunction(
                        name=tc.name,
                        arguments=json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    ),
                )
                for tc in result.tool_calls
            ]

        # Determine content based on structured output or plain text
        content = result.content
        if result.structured and result.structured.success:
            # If structured output was requested and successful, use parsed content
            # Note: SmolAgents might expect specific formatting for structured responses
            # For now, we'll keep the raw content and let the agent handle parsing if needed
            # Or we could serialize the parsed object back to string
            pass

        # Create usage info
        token_usage = None
        if result.usage:
            token_usage = TokenUsage(
                input_tokens=result.usage.get("prompt_tokens", 0),
                output_tokens=result.usage.get("completion_tokens", 0),
            )

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            raw=result,
            token_usage=token_usage,
        )
