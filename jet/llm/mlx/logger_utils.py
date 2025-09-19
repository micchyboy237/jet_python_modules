import os
from datetime import datetime, UTC
from typing import Any, Literal, Optional, Union, List
from uuid import uuid4

from jet.file.utils import save_file
from jet.models.model_types import CompletionResponse, Message
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import get_method_info
from jet._token.token_utils import token_counter

from shared.setup.events import EventSettings


ALLOWED_METHODS = {"chat", "stream_chat", "generate", "stream_generate"}


def get_next_file_counter(log_dir: str, method: str) -> str:
    """Generate a timestamp-based prefix for the log file."""
    return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


class ChatLogger:
    """Handles logging of chat interactions to a specified directory."""

    def __init__(
        self,
        log_dir: str,
        method: Literal["chat", "stream_chat", "generate", "stream_generate"] = "chat",
        limit: Optional[int] = None
    ):
        if method not in ALLOWED_METHODS:
            raise ValueError(
                f"Invalid method '{method}'. Allowed methods are: {sorted(ALLOWED_METHODS)}"
            )

        start_time = EventSettings.get_entry_time()
        # Format start_time as "YYYYMMDD_HHMMSS" (no colons)
        dt = datetime.fromisoformat(start_time)
        formatted_start_time = dt.strftime("%Y%m%d_%H%M%S")

        self.log_dir = os.path.join(log_dir, formatted_start_time)
        self.method = method
        self.limit = limit
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)

    def log_interaction(
        self,
        messages: Union[str, List[Message]],
        response: Union[str, CompletionResponse, List[CompletionResponse]],
        model: str,
        method: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log prompt or messages and response to a timestamped file with additional metadata."""
        effective_method = method if method is not None else self.method

        if effective_method not in ALLOWED_METHODS:
            raise ValueError(
                f"Invalid method '{effective_method}'. Allowed methods are: {sorted(ALLOWED_METHODS)}"
            )

        timestamp_prefix = get_next_file_counter(self.log_dir, effective_method)
        filename = f"{timestamp_prefix}_{effective_method}.json"
        log_file = os.path.join(self.log_dir, filename)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        tools = None
        if "tools" in kwargs:
            tools = (kwargs.pop("tools") or []).copy()
            for tool_idx, tool_fn in enumerate(tools):
                if callable(tool_fn):
                    tools[tool_idx] = get_method_info(tool_fn)

        prompt_tokens = token_counter(messages, model)
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "method": effective_method,
            "tools": tools,
        }
        log_data.update(kwargs)

        if isinstance(messages, str):
            log_data["prompt"] = messages
        else:
            log_data["messages"] = messages.copy()

        if isinstance(response, str):
            log_data["response"] = response
        elif isinstance(response, (list, dict)):
            resp_copy = response.copy() if isinstance(response, dict) else response
            if isinstance(resp_copy, dict) and "usage" in resp_copy:
                usage = resp_copy.pop("usage")
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
                resp_copy["usage"] = formatted_usage
            if isinstance(resp_copy, dict) and "choices" in resp_copy:
                choices = resp_copy.pop("choices")
                resp_copy["choices"] = choices
            log_data["response"] = format_json(resp_copy, indent=2)
        else:
            log_data["response"] = str(response)

        save_file(log_data, log_file)

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
