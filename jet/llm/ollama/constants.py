"""Set of constants."""

import os

DEFAULT_TEMPERATURE = 0.1
DEFAULT_CONTEXT_WINDOW = 4096  # tokens
DEFAULT_NUM_OUTPUTS = 256  # tokens
DEFAULT_NUM_INPUT_FILES = 10  # files
DEFAULT_REQUEST_TIMEOUT = 300.0

DEFAULT_EMBED_BATCH_SIZE = 32

DEFAULT_SIMILARITY_TOP_K = 2
DEFAULT_IMAGE_SIMILARITY_TOP_K = 2

# NOTE: for text-embedding-ada-002
DEFAULT_EMBEDDING_DIM = 1536

# context window size for llm predictor
COHERE_CONTEXT_WINDOW = 2048
AI21_J2_CONTEXT_WINDOW = 8192


TYPE_KEY = "__type__"
DATA_KEY = "__data__"
VECTOR_STORE_KEY = "vector_store"
IMAGE_STORE_KEY = "image_store"
GRAPH_STORE_KEY = "graph_store"
INDEX_STORE_KEY = "index_store"
DOC_STORE_KEY = "doc_store"
PG_STORE_KEY = "property_graph_store"

# llama-cloud constants
DEFAULT_PIPELINE_NAME = "default"
DEFAULT_PROJECT_NAME = "Default"
DEFAULT_BASE_URL = "https://api.cloud.llamaindex.ai"
DEFAULT_APP_URL = "https://cloud.llamaindex.ai"

# Ollama constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_LLM_URL", "http://localhost:11434")
OLLAMA_BASE_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434")
OLLAMA_LARGE_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepscaler")
OLLAMA_SMALL_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepscaler")

OLLAMA_SMALL_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
OLLAMA_SMALL_CHUNK_SIZE = 500  # tokens
OLLAMA_SMALL_CHUNK_OVERLAP = 50  # tokens

OLLAMA_LARGE_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
OLLAMA_LARGE_CHUNK_SIZE = 1000  # tokens
OLLAMA_LARGE_CHUNK_OVERLAP = 100  # tokens

DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepscaler")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "deepscaler")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe")
