# jet.adapters.llama_cpp.config

import os

LLM_BASE_URL = os.getenv("LLAMA_CPP_LLM_URL")
LLM_MODEL = os.getenv("LLAMA_CPP_LLM_MODEL")
EMBED_MODEL = os.getenv("LLAMA_CPP_EMBED_MODEL")
EMBED_BASE_URL = os.getenv("LLAMA_CPP_EMBED_URL")
EMBED_DIMS = int(os.getenv("LLAMA_CPP_EMBED_DIMS", "768"))
