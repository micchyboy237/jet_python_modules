from typing import Optional, Dict, Any
from jet.token.token_utils import token_counter
from langchain_ollama import ChatOllama as BaseChatOllama
from jet.logger import logger

from shared.events import EventSettings


class ChatOllama(BaseChatOllama):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Generate headers dynamically
        event = EventSettings.call_ollama_chat_langchain()
        pre_start_hook_start_time = EventSettings.event_data["pre_start_hook"]["start_time"]
        log_filename = event["filename"].split(".")[0]
        logger.log("Log-Filename:", log_filename, colors=["WHITE", "DEBUG"])
        token_count = token_counter(kwargs.get(
            "messages", []), model=model) if kwargs.get("messages") else 0
        headers = {
            "Tokens": str(token_count),
            "Log-Filename": log_filename,
            "Event-Start-Time": pre_start_hook_start_time,
        }

        # Combine provided `client_kwargs` with default headers
        client_kwargs = client_kwargs or {}
        client_kwargs.setdefault("headers", headers)

        # Call the parent class initializer with updated parameters
        super().__init__(model=model, base_url=base_url,
                         client_kwargs=client_kwargs, **kwargs)
