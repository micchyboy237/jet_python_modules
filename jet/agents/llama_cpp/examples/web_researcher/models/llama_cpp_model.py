# web_researcher/models/llama_cpp_model.py
"""
Custom llama.cpp model wrapper for smolagents with context window management.
"""

import json
import logging
from typing import Dict, List, Optional

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL
from jet.adapters.llama_cpp.model_utils import get_llama_cpp_base_url
from openai import OpenAI
from smolagents.models import Model

logger = logging.getLogger(__name__)


class LlamaCppModel(Model):
    """
    Model wrapper for llama.cpp server with context window management.
    Tracks token usage and truncates history if needed.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
        max_context_length: Optional[int] = 4096,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        **kwargs,
    ):
        """
        Initialize llama.cpp model.

        Args:
            model_id: Model identifier (defaults to LLM_MODEL from env)
            base_url: llama.cpp server URL (defaults to LLM_BASE_URL)
            max_context_length: Maximum context window size
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
        """
        self.model_id = model_id or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL or "http://localhost:8080"
        self.base_url = get_llama_cpp_base_url(override=self.base_url)
        self.max_context_length = max_context_length
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Initialize OpenAI client for llama.cpp compatibility
        self.client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="not-needed",
            timeout=60.0,
            max_retries=2,
        )

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(
            f"Initialized LlamaCppModel with model={self.model_id}, url={self.base_url}"
        )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate completion from llama.cpp server.

        Args:
            messages: Chat messages list
            stop_sequences: Sequences to stop generation
            grammar: JSON grammar for structured output
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Generated text response
        """
        # Check context window and truncate if needed
        messages = self._truncate_messages(messages)

        # Prepare request
        request_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            "stream": False,
        }

        # Add stop sequences if provided
        if stop_sequences:
            request_kwargs["stop"] = stop_sequences

        # Add grammar for structured output (if supported)
        if grammar:
            request_kwargs["grammar"] = grammar

        try:
            # Make API call
            logger.debug(f"Sending request with {len(messages)} messages")
            response = self.client.chat.completions.create(**request_kwargs)

            # Track token usage
            if hasattr(response, "usage"):
                usage = response.usage
                self.total_input_tokens += usage.prompt_tokens or 0
                self.total_output_tokens += usage.completion_tokens or 0

            result = response.choices[0].message.content or ""

            # Handle tool calls if present
            if hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    logger.info(f"Received {len(tool_calls)} tool calls")
                    # Return as JSON string for ToolCallingAgent
                    result = json.dumps(
                        {
                            "tool_calls": [
                                {
                                    "name": tc.function.name,
                                    "arguments": json.loads(tc.function.arguments)
                                    if tc.function.arguments
                                    else {},
                                }
                                for tc in tool_calls
                            ]
                        }
                    )

            return result

        except Exception as e:
            logger.error(f"Error calling llama.cpp: {e}")
            # Return error message that agent can handle
            return f"Error: {str(e)}"

    def _truncate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Truncate messages to fit within context window.
        Keeps system prompt and most recent messages.

        Args:
            messages: Original messages list

        Returns:
            Truncated messages list
        """
        if not self.max_context_length:
            return messages

        # Estimate token count (rough: 4 chars per token)
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.max_context_length:
            return messages

        logger.warning(
            f"Messages estimated {estimated_tokens} tokens, truncating to {self.max_context_length}"
        )

        # Keep system message, truncate from the oldest messages
        result = []
        system_msg = None
        other_msgs = []

        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg
            else:
                other_msgs.append(msg)

        # Keep most recent messages
        result = [system_msg] if system_msg else []
        current_length = sum(len(m.get("content", "")) for m in result)

        # Add messages from newest to oldest until we hit limit
        for msg in reversed(other_msgs):
            msg_len = len(msg.get("content", ""))
            if current_length + msg_len > self.max_context_length * 4:
                # Truncate this message
                content = msg.get("content", "")
                truncated = content[
                    : (self.max_context_length * 4 - current_length - 50)
                ]
                result.append({"role": msg["role"], "content": truncated + "..."})
                break
            result.append(msg)
            current_length += msg_len

        # Reverse back to chronological order
        return [result[0]] + result[1:][::-1] if result else result

    def get_token_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

    def reset_token_stats(self) -> None:
        """Reset token statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
