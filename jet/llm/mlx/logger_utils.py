import os
from datetime import datetime, UTC
from typing import Any, Literal, Optional, Union, List
from uuid import uuid4

from jet.file.utils import save_file
from jet.models.model_types import CompletionResponse, Message
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import get_method_info

from shared.setup.events import EventSettings


def get_next_file_counter(log_dir: str, method: str) -> str:
    """Generate a timestamp-based prefix for the log file."""
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


class ChatLogger:
    """Handles logging of chat interactions to a specified directory."""

    def __init__(
        self,
        log_dir: str,
        method: Literal["chat", "stream_chat", "generate", "stream_generate"],
        limit: Optional[int] = None
    ):
        start_time = EventSettings.get_entry_event()["start_time"]
        # Format start_time as "YYYY-MM-DD|HH:MM:SS"
        dt = datetime.fromisoformat(start_time)
        formatted_start_time = dt.strftime("%Y-%m-%d|%H:%M:%S")

        self.log_dir = os.path.join(log_dir, formatted_start_time)
        self.method = method
        self.limit = limit
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

    def log_interaction(
        self,
        prompt_or_messages: Union[str, List[Message]],
        response: Union[CompletionResponse, List[CompletionResponse]],
        **kwargs: Any
    ) -> None:
        """Log prompt or messages and response to a timestamped file with additional metadata."""
        timestamp_prefix = get_next_file_counter(self.log_dir, self.method)
        filename = f"{timestamp_prefix}_{self.method}.json"
        log_file = os.path.join(self.log_dir, filename)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        tools = None
        if "tools" in kwargs:
            tools = (kwargs.pop("tools") or []).copy()
            for tool_idx, tool_fn in enumerate(tools):
                if callable(tool_fn):
                    tools[tool_idx] = get_method_info(tool_fn)

        # Initialize log_data with core attributes
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "method": self.method,
            "tools": tools,
        }

        # Add remaining kwargs (excluding usage for now)
        log_data.update(kwargs)

        # Handle prompt or messages before response
        if isinstance(prompt_or_messages, str):
            log_data["prompt"] = prompt_or_messages
        else:
            log_data["messages"] = prompt_or_messages.copy()

            # Add assistant role message with response text
            if isinstance(response, str):
                response_text = response
            elif isinstance(response, list):
                response_text = "\n".join([r["content"] for r in response])
            else:
                response_text = response["content"]

            log_data["messages"].append({
                "role": "assistant",
                "content": response_text
            })

        # Handle usage formatting if present
        formatted_usage = None
        if "usage" in response:
            usage = response.pop("usage")
            formatted_usage = {}
            if "prompt_tokens" in usage:
                formatted_usage["prompt_tokens"] = usage["prompt_tokens"]
            if "prompt_tps" in usage:
                val = usage["prompt_tps"]
                formatted_usage["prompt_tps"] = f"{val:.2f} tokens/sec" if isinstance(val, float) else val
            if "completion_tokens" in usage:
                formatted_usage["completion_tokens"] = usage["completion_tokens"]
            if "completion_tps" in usage:
                val = usage["completion_tps"]
                formatted_usage["completion_tps"] = f"{val:.2f} tokens/sec" if isinstance(val, float) else val
            if "peak_memory" in usage:
                val = usage["peak_memory"]
                formatted_usage["peak_memory"] = f"{val:.2f} GB" if isinstance(val, float) else val
            if "total_tokens" in usage:
                formatted_usage["total_tokens"] = usage["total_tokens"]
            response["usage"] = formatted_usage

        if "choices" in response:
            choices = response.pop("choices")
            response["choices"] = choices

        # Add response last
        log_data["response"] = format_json(response, indent=2)

        save_file(log_data, log_file)

        # Enforce file limit
        if self.limit is not None:
            self._enforce_limit()

    def _enforce_limit(self) -> None:
        """Remove oldest files if log count exceeds the specified limit."""
        files = sorted(
            (f for f in os.listdir(self.log_dir) if f.endswith(".json")),
            key=lambda f: f.split('_')[0]  # Sort by timestamp prefix
        )

        excess = len(files) - self.limit
        if excess > 0:
            for old_file in files[:excess]:
                os.remove(os.path.join(self.log_dir, old_file))
