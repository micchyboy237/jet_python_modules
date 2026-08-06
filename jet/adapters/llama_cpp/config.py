# jet.adapters.llama_cpp.config

import os

DEFAULT_LLM_MODEL = "qwen3.5-uncensored:2b"
DEFAULT_EMBED_MODEL = "nomic-embed:2-moe"
DEFAULT_EMBED_LG_MODEL = "nomic-embed:1.5"

# e.g. "http://192.168.68.150:8080"
LLM_BASE_HOST = os.getenv("LLAMA_CPP_LLM_HOST")

# e.g. "http://192.168.68.150:8080/v1"
LLM_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL")

# e.g. "qwen3.5-uncensored:2b"
LLM_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL", DEFAULT_LLM_MODEL)


# e.g. "nomic-embed:2-moe"
EMBED_MODEL = os.getenv("LLAMA_CPP_EMBED_MODEL", DEFAULT_EMBED_MODEL)

# e.g. "http://192.168.68.150:8081"
EMBED_BASE_HOST = os.getenv("LLAMA_CPP_EMBED_HOST")

# e.g. "http://192.168.68.150:8081/v1"
EMBED_BASE_URL = os.getenv("LLAMA_CPP_EMBED_URL")

# e.g. 768
EMBED_DIMS = int(os.getenv("LLAMA_CPP_EMBED_DIMS", "768"))

# e.g. "search_query: "
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "")

# e.g. "search_document: "
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "")


EMBED_LG_MODEL = os.getenv("LLAMA_CPP_EMBED_LG_MODEL", DEFAULT_EMBED_LG_MODEL)
EMBED_LG_BASE_HOST = os.getenv("LLAMA_CPP_EMBED_LG_HOST")
EMBED_LG_BASE_URL = os.getenv("LLAMA_CPP_EMBED_LG_URL")
EMBED_LG_DIMS = int(os.getenv("LLAMA_CPP_EMBED_LG_DIMS", "768"))
EMBED_LG_QUERY_PREFIX = os.getenv("EMBED_LG_QUERY_PREFIX", "")
EMBED_LG_DOC_PREFIX = os.getenv("EMBED_LG_DOC_PREFIX", "")


# e.g. "bge-rerank:v2-m3"
RERANK_MODEL = os.getenv("LLAMA_CPP_RERANK_MODEL")

# e.g. "http://192.168.68.150:8082"
RERANK_BASE_HOST = os.getenv("LLAMA_CPP_RERANK_HOST")

# e.g. "http://192.168.68.150:8082/v1"
RERANK_BASE_URL = os.getenv("LLAMA_CPP_RERANK_URL")

# e.g. 1024
RERANK_DIMS = int(os.getenv("LLAMA_CPP_RERANK_DIMS", "1024"))


# Vision Model
VISION_MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", "qwen3.5-uncensored:2b")
VISION_HF_MODEL = os.getenv("LLAMA_CPP_VISION_HF_MODEL", "Qwen/Qwen3.5-2B")
VISION_BASE_HOST = os.getenv("LLAMA_CPP_VISION_HOST")
VISION_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL")
