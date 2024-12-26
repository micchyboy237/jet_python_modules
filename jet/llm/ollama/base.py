from typing import Optional
from llama_index.core.callbacks.base import CallbackManager
from llama_index.llms.ollama import Ollama as BaseOllama
from llama_index.embeddings.ollama import OllamaEmbedding as BaseOllamaEmbedding
from jet.llm.ollama import (
    large_embed_model,
    DEFAULT_LLM_SETTINGS,
    DEFAULT_EMBED_SETTINGS,
)
from jet.logger import logger


class Ollama(BaseOllama):
    """
    Extends functionality of BaseOllama.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_SETTINGS["model"],
        context_window: int = DEFAULT_LLM_SETTINGS["context_window"],
        request_timeout: float = DEFAULT_LLM_SETTINGS["request_timeout"],
        temperature: float = DEFAULT_LLM_SETTINGS["temperature"],
        base_url: str = DEFAULT_LLM_SETTINGS["base_url"],
        **kwargs
    ):
        # Passing parameters to parent class constructor
        super().__init__(
            model=model,
            context_window=context_window,
            request_timeout=request_timeout,
            temperature=temperature,
            base_url=base_url,
            **kwargs
        )

    def __call__(self, *args, **kwargs) -> None:
        pass


class OllamaEmbedding(BaseOllamaEmbedding):
    """
    Extends functionality of BaseOllamaEmbedding.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_SETTINGS["model_name"],
        base_url: str = DEFAULT_EMBED_SETTINGS["base_url"],
        embed_batch_size: int = DEFAULT_EMBED_SETTINGS['embed_batch_size'],
        ollama_additional_kwargs: dict[str,
                                       any] = DEFAULT_EMBED_SETTINGS['ollama_additional_kwargs'],
        callback_manager: Optional[CallbackManager] = None,
        num_workers: Optional[int] = None,
    ):
        # Passing parameters to parent class constructor
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=embed_batch_size,
            ollama_additional_kwargs=ollama_additional_kwargs,
            callback_manager=callback_manager,
            num_workers=num_workers,
        )

    def embed_documents(
        self,
        texts: list[str],
        key: str = ""
    ) -> Optional[list[list[float]]]:
        import requests

        try:
            r = requests.post(
                f"{self.base_url}/api/embed",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json={"input": texts, "model": self.model_name},
            )
            r.raise_for_status()
            data = r.json()

            if "embeddings" in data:
                logger.log("Embed Token Count:", data["prompt_eval_count"], colors=[
                           "WHITE", "SUCCESS"])
                return data["embeddings"]
            else:
                raise ValueError("Something went wrong :/")
        except Exception as e:
            logger.error(e)
            return None
