from typing import TypedDict
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# LLM and embedding config

base_url = "http://localhost:11434"
small_llm_model = "llama3.2"
large_llm_model = "llama3.1"
small_embed_model = "nomic-embed-text"
large_embed_model = "mxbai-embed-large"
DEFAULT_LLM_SETTINGS = {
    "model": large_llm_model,
    "context_window": 4096,
    "request_timeout": 300.0,
    "temperature": 0,
}
DEFAULT_EMBED_SETTINGS = {
    "model": small_embed_model,
    "chunk_size": 768,
    "chunk_overlap": 75,
}


class SettingsDict(TypedDict, total=False):
    llm_model: str
    context_window: int
    request_timeout: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str


def update_llm_settings(settings: SettingsDict = {}):
    if settings.get("chunk_size"):
        Settings.chunk_size = settings["chunk_size"]

    if settings.get("chunk_overlap"):
        Settings.chunk_overlap = settings["chunk_overlap"]

    if settings.get("embedding_model"):
        Settings.embed_model = create_embed_model(
            model=settings["embedding_model"],
            base_url=settings.get("base_url", base_url),
        )

    if settings.get("llm_model"):
        Settings.llm = create_llm(
            model=settings["llm_model"],
            base_url=settings.get("base_url", base_url),
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


def create_embed_model(model: str = DEFAULT_EMBED_SETTINGS['model'], base_url: str = base_url):
    embed_model = OllamaEmbedding(
        model_name=model,
        base_url=base_url,
    )
    Settings.embed_model = embed_model
    return embed_model
