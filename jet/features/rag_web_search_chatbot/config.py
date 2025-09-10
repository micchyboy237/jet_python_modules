import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """Retrieve environment variable with fallback to default."""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


# Configuration variables with previous defaults
TAVILY_API_KEY = get_env_var("TAVILY_API_KEY")
OLLAMA_BASE_URL = get_env_var("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = get_env_var("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = get_env_var("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIRECTORY = get_env_var(
    "CHROMA_PERSIST_DIRECTORY", "./chroma_db")
DOCS_DIRECTORY = get_env_var("DOCS_DIRECTORY", "docs")
CHUNK_SIZE = int(get_env_var("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(get_env_var("CHUNK_OVERLAP", "200"))
RETRIEVER_K = int(get_env_var("RETRIEVER_K", "3"))

# Generation parameters with previous defaults
GENERATION_PARAMS = {
    "temperature": float(get_env_var("TEMPERATURE", "0.0")),
    "max_tokens": int(get_env_var("MAX_TOKENS", "500")),
    "top_p": float(get_env_var("TOP_P", "1.0")),
    "top_k": int(get_env_var("TOP_K", "40")),
}
