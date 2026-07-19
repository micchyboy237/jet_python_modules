from typing import Optional, Union

from jet.adapters.keybert.embeddings import KeyBERTLlamacppEmbedder
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.logger import logger

from keybert import KeyBERT as BaseKeyBERT
from keybert.backend import BaseEmbedder
from keybert.llm._base import BaseLLM

DEFAULT_EMBEDDING_MODEL: str = EMBED_MODEL


class KeyBERT(BaseKeyBERT):
    def __init__(
        self,
        model: Union[str, LLAMACPP_EMBED_KEYS, BaseEmbedder] = None,
        llm: Optional[BaseLLM] = None,
    ):
        target_model = model or DEFAULT_EMBEDDING_MODEL

        if isinstance(target_model, str):
            logger.info(
                f"Initializing KeyBERTLlamacppEmbedder adapter wrapper for model: {target_model}"
            )
            embedder = KeyBERTLlamacppEmbedder(target_model)
            target_model = embedder

        super().__init__(
            model=target_model,
            llm=llm,
        )
