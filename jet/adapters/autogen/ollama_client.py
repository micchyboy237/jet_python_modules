from autogen_ext.models.ollama import OllamaChatCompletionClient as BaseOllamaChatCompletionClient
from jet.llm.mlx.logger_utils import ChatLogger
from jet.llm.mlx.config import DEFAULT_OLLAMA_LOG_DIR
from jet.logger import logger


class OllamaChatCompletionClient(BaseOllamaChatCompletionClient):
    async def create(self, *args, **kwargs):
        result = await super().create(*args, **kwargs)

        # Log to console
        content = result.content if isinstance(
            result.content, str) else str(result.content)
        logger.teal(content)

        # Log to file
        ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method="chat").log_interaction(
            kwargs.get("messages", []),
            result.model_dump(),
            model=self._model_name,
            tools=kwargs.get("tools"),
        )
        return result

    async def create_stream(self, *args, **kwargs):
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
                ChatLogger(DEFAULT_OLLAMA_LOG_DIR, method="stream_chat").log_interaction(
                    kwargs.get("messages", []),
                    chunk.model_dump(),
                    model=self._model_name,
                    tools=kwargs.get("tools"),
                )
                yield chunk
