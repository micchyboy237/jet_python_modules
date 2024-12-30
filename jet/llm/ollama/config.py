from typing import TypedDict
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# LLM and embedding config

base_url = "http://localhost:11434"
base_embed_url = "http://localhost:11434"
small_llm_model = "llama3.2"
large_llm_model = "llama3.1"
small_embed_model = "mxbai-embed-large"
large_embed_model = "nomic-embed-text"
DEFAULT_LLM_SETTINGS = {
    "model": large_llm_model,
    "context_window": 4096,
    "request_timeout": 300.0,
    "temperature": 0,
    "base_url": base_url,
}
DEFAULT_EMBED_SETTINGS = {
    "model_name": large_embed_model,
    "base_url": base_embed_url,
    "embed_batch_size": 10,
    "ollama_additional_kwargs": {}
}


class SettingsDict(TypedDict, total=False):
    llm_model: str
    context_window: int
    request_timeout: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


class EnhancedSettings(Settings):
    def __init__(self, settings: dict = {}):
        super().__init__()

        # Initialize settings based on input or default values
        self.embed_model = create_embed_model(
            model=settings.get("embedding_model",
                               DEFAULT_EMBED_SETTINGS['model_name']),
            base_url=settings.get(
                "base_url", DEFAULT_EMBED_SETTINGS['base_url']),
        )

        llm_model = settings.get("llm_model", DEFAULT_LLM_SETTINGS['model'])
        self.llm = create_llm(
            model=llm_model,
            base_url=settings.get(
                "base_url", DEFAULT_LLM_SETTINGS['base_url']),
            temperature=settings.get(
                "temperature", DEFAULT_LLM_SETTINGS['temperature']),
            context_window=settings.get(
                "context_window", DEFAULT_LLM_SETTINGS['context_window']),
            request_timeout=settings.get(
                "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
        )

        self.chunk_size = settings.get("chunk_size", None)
        self.chunk_overlap = settings.get("chunk_overlap", None)

    def count_tokens(self, text: str) -> int:
        from jet.llm.token import token_counter
        return token_counter(text, self.llm.model)


def initialize_ollama_settings(settings: SettingsDict = {}):
    EnhancedSettings.embed_model = create_embed_model(
        model=settings.get("embedding_model",
                           DEFAULT_EMBED_SETTINGS['model_name']),
        base_url=settings.get("base_url", DEFAULT_EMBED_SETTINGS['base_url']),
    )

    llm_model = settings.get("llm_model", DEFAULT_LLM_SETTINGS['model'])
    EnhancedSettings.llm = create_llm(
        model=llm_model,
        base_url=settings.get("base_url", DEFAULT_LLM_SETTINGS['base_url']),
        temperature=settings.get(
            "temperature", DEFAULT_LLM_SETTINGS['temperature']),
        context_window=settings.get(
            "context_window", DEFAULT_LLM_SETTINGS['context_window']),
        request_timeout=settings.get(
            "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
    )

    if settings.get("chunk_size"):
        EnhancedSettings.chunk_size = settings["chunk_size"]

    if settings.get("chunk_overlap"):
        EnhancedSettings.chunk_overlap = settings["chunk_overlap"]

    return EnhancedSettings


def update_llm_settings(settings: SettingsDict = {}):
    if settings.get("chunk_size"):
        Settings.chunk_size = settings["chunk_size"]

    if settings.get("chunk_overlap"):
        Settings.chunk_overlap = settings["chunk_overlap"]

    if settings.get("embedding_model"):
        Settings.embed_model = create_embed_model(
            model=settings.get("embedding_model",
                               DEFAULT_EMBED_SETTINGS['model_name']),
            base_url=settings.get(
                "base_url", DEFAULT_EMBED_SETTINGS['base_url']),
        )

    if settings.get("llm_model"):
        Settings.llm = create_llm(
            model=settings.get("llm_model", DEFAULT_LLM_SETTINGS['model']),
            base_url=settings.get(
                "base_url", DEFAULT_LLM_SETTINGS['base_url']),
            temperature=settings.get(
                "temperature", DEFAULT_LLM_SETTINGS['temperature']),
            context_window=settings.get(
                "context_window", DEFAULT_LLM_SETTINGS['context_window']),
            request_timeout=settings.get(
                "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
        )

    return Settings


def create_llm(
    model: str = DEFAULT_LLM_SETTINGS['model'],
    base_url: str = base_url,
    temperature: float = DEFAULT_LLM_SETTINGS['temperature'],
    context_window: int = DEFAULT_LLM_SETTINGS['context_window'],
    request_timeout: float = DEFAULT_LLM_SETTINGS['request_timeout'],
):
    llm = Ollama(
        temperature=temperature,
        context_window=context_window,
        request_timeout=request_timeout,
        model=model,
        base_url=base_url,
    )
    Settings.llm = llm
    return llm


def create_embed_model(
    model: str = DEFAULT_EMBED_SETTINGS['model_name'],
    base_url: str = base_url,
    embed_batch_size: int = DEFAULT_EMBED_SETTINGS['embed_batch_size'],
    ollama_additional_kwargs: dict[str,
                                   any] = DEFAULT_EMBED_SETTINGS['ollama_additional_kwargs'],
):
    embed_model = OllamaEmbedding(
        model_name=model,
        base_url=base_url,
        embed_batch_size=embed_batch_size,
        ollama_additional_kwargs=ollama_additional_kwargs,
    )
    Settings.embed_model = embed_model
    return embed_model
