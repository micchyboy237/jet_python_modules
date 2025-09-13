from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.logger import logger
from jet.transformers.formatters import format_json

DETERMINISTIC_LLM_SETTINGS = {
    "seed": 42,
    "temperature": 0,
    "num_keep": 0,
    "num_predict": -1,
}


class OllamaChatCompletionClient(BaseOllamaChatCompletionClient):
    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 300.0, options: dict = None, **kwargs):
        # Use DETERMINISTIC_LLM_SETTINGS as default options
        options = {**DETERMINISTIC_LLM_SETTINGS, **(options or {})}
        super().__init__(model=model, host=host, timeout=timeout, options=options, **kwargs)

    async def create(self, *args, **kwargs):
        logger.gray("Chat LLM Settings:")
        logger.info(format_json({
            "args": args,
            "kwargs": kwargs,
        }))

        # Use create_stream to handle the request
        result = None
        async for chunk in self.create_stream(*args, **kwargs, method="chat"):
            result = chunk  # Store the last chunk as the final result

        return result

    async def create_stream(self, *args, method: str = "stream_chat", **kwargs):
        logger.gray("Stream Chat LLM Settings:")
        logger.info(format_json({
            "args": args,
            "kwargs": kwargs,
        }))

        async for chunk in super().create_stream(*args, **kwargs):
            if isinstance(chunk, str):
                # Log partial chunks to console
                logger.teal(chunk, flush=True)
                yield chunk
            else:
                # Log final result to console
                content = chunk.content if isinstance(
                    chunk.content, str) else str(chunk.content)
                logger.teal(content, flush=True)

                # Log to file
                ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method=method).log_interaction(
                    args[0],
                    chunk.model_dump(),
                    model=self._model_name,
                    tools=kwargs.get("tools"),
                )
                yield chunk
