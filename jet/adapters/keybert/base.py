from typing import Optional, Union
from keybert import KeyBERT as BaseKeyBERT
from keybert.backend import BaseEmbedder
from jet.adapters.keybert.embeddings import KeyBERTLlamacppEmbedder
from keybert.llm._base import BaseLLM

from jet.llm.models import OLLAMA_MODEL_NAMES

DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

class KeyBERT(BaseKeyBERT):
    def __init__(
        self,
        model: Union[str, OLLAMA_MODEL_NAMES, BaseEmbedder] = DEFAULT_EMBEDDING_MODEL,
        llm: Optional[BaseLLM] = None,
        use_cache: bool = False,
    ):
        if not model or isinstance(model, str):
            embedder = KeyBERTLlamacppEmbedder(model or DEFAULT_EMBEDDING_MODEL, use_cache=use_cache)
            model = embedder

        super().__init__(
            model=model,
            llm=llm,
        )
