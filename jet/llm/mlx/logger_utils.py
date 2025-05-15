import os
from datetime import datetime, UTC
from typing import Any, Literal, Union, List
from uuid import uuid4

from jet.file.utils import save_file
from jet.llm.mlx.mlx_types import CompletionResponse, Message
from jet.transformers.formatters import format_json


def short_sortable_filename() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")  # e.g., '20250515T142311'
    suffix = uuid4().hex[:4]                          # e.g., 'a3f9'
    return f"{ts}_{suffix}"                           # '20250515T142311_a3f9'


class ChatLogger:
    """Handles logging of chat interactions to a specified directory."""

    def __init__(
        self,
        log_dir: str,
        method: Literal["chat", "stream_chat", "generate", "stream_generate"],
        limit: int = 15
    ):
        self.log_dir = log_dir
        self.method = method
        self.limit = limit
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log_interaction(
        self,
        prompt_or_messages: Union[str, List[Message]],
        response: Union[CompletionResponse, List[CompletionResponse]],
        **kwargs: Any
    ) -> None:
        """Log prompt or messages and response to a timestamped file with additional metadata."""
        filename = f"{short_sortable_filename()}_{self.method}.json"
        log_file = os.path.join(self.log_dir, filename)

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "method": self.method,
        }

        if isinstance(prompt_or_messages, str):
            log_data["prompt"] = prompt_or_messages
        else:
            log_data["messages"] = prompt_or_messages

        log_data["response"] = format_json(response, indent=2)

        # Move usage to the end and format as readable strings
        if "usage" in kwargs:
            usage = kwargs.pop("usage")
            formatted_usage = {
                "prompt_tokens": str(usage.get("prompt_tokens", 0)),
                "prompt_tps": f"{usage.get('prompt_tps', 0):.2f} tokens/sec",
                "completion_tokens": str(usage.get("completion_tokens", 0)),
                "completion_tps": f"{usage.get('completion_tps', 0):.2f} tokens/sec",
                "peak_memory": f"{usage.get('peak_memory', 0):.2f} GB",
                "total_tokens": str(usage.get("total_tokens", 0))
            }
            kwargs["usage"] = formatted_usage

        log_data.update(kwargs)  # Add remaining kwargs, with usage last

        save_file(log_data, log_file)

        # Enforce file limit
        if self.limit is not None:
            self._enforce_limit()

    def _enforce_limit(self) -> None:
        """Remove oldest files if log count exceeds the specified limit."""
        files = sorted(
            (f for f in os.listdir(self.log_dir) if f.endswith(".json")),
            key=lambda f: f
        )

        excess = len(files) - self.limit
        if excess > 0:
            for old_file in files[:excess]:
                os.remove(os.path.join(self.log_dir, old_file))
