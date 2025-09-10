from langchain_ollama import OllamaEmbeddings as BaseOllamaEmbeddings


class OllamaEmbeddings(BaseOllamaEmbeddings):
    def __init__(self, **kwargs):
        # Set default model to "all-minilm:33m" if not provided
        kwargs.setdefault("model", "all-minilm:33m")
        super().__init__(**kwargs)
