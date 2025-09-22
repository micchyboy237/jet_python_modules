from typing import Dict
from jet.llm.ollama.base import OllamaEmbedding

def get_ollama_embed_model(model: str = "nomic-embed-text", base_url: str = "http://localhost:11435", batch_size: int = 32, ollama_additional_kwargs: Dict = {}) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=model,
        base_url=base_url,
        embed_batch_size=batch_size,
        ollama_additional_kwargs=ollama_additional_kwargs,
    )