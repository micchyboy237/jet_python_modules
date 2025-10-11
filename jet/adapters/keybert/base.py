from keybert import KeyBERT as BaseKeyBERT
from jet.adapters.keybert.embeddings import KeyBERTLlamacppEmbedder
from keybert.llm._base import BaseLLM

DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

class KeyBERT(BaseKeyBERT):
    def __init__(
        self,
        model=DEFAULT_EMBEDDING_MODEL,
        llm: BaseLLM = None,
    ):
        if not model or isinstance(model, str):
            embedder = KeyBERTLlamacppEmbedder(model or DEFAULT_EMBEDDING_MODEL)
            model = embedder

        super().__init__(
            model=model,
            llm=llm,
        )
