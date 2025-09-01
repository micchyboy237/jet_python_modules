import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient


class OllamaChatCompletionClient(BaseOllamaChatCompletionClient):
    def __init__(self, *args, log_dir: str | None = None, **kwargs):
        """
        Extension of BaseOllamaChatCompletionClient with optional request/response logging.

        Args:
            log_dir (str | None): Directory to store request/response logs. If None, logging is disabled.
        """
        super().__init__(*args, **kwargs)

        self._log_dir = log_dir
        self._logger = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, "ollama_client.log")
            handler = RotatingFileHandler(
                log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)

            self._logger = logging.getLogger(f"OllamaClientLogger_{id(self)}")
            self._logger.setLevel(logging.INFO)
            self._logger.addHandler(handler)
            self._logger.propagate = False

    def _log_request(self, endpoint: str, payload: dict):
        if not self._logger:
            return
        try:
            self._logger.info(
                "REQUEST %s\n%s",
                endpoint,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )
        except Exception as e:
            self._logger.error(f"Failed to log request: {e}")

    def _log_response(self, endpoint: str, response: dict):
        if not self._logger:
            return
        try:
            self._logger.info(
                "RESPONSE %s\n%s",
                endpoint,
                json.dumps(response, indent=2, ensure_ascii=False),
            )
        except Exception as e:
            self._logger.error(f"Failed to log response: {e}")

    async def create(self, *args, **kwargs):
        # Log request before sending
        self._log_request("create", {"args": args, "kwargs": kwargs})

        result = await super().create(*args, **kwargs)

        # Log response after receiving
        try:
            self._log_response("create", result.model_dump())
        except Exception as e:
            if self._logger:
                self._logger.error(f"Failed to log response: {e}")

        return result

    async def create_stream(self, *args, **kwargs):
        # Log request
        self._log_request("create_stream", {"args": args, "kwargs": kwargs})

        async for chunk in super().create_stream(*args, **kwargs):
            try:
                if hasattr(chunk, "model_dump"):
                    self._log_response("create_stream", chunk.model_dump())
                else:
                    self._log_response("create_stream", {"chunk": str(chunk)})
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Failed to log streaming chunk: {e}")
            yield chunk
