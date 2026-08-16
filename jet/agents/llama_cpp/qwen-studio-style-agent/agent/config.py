import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    LLAMA_BASE_URL: str = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")
    LLAMA_MODEL: str = os.getenv("LLAMA_MODEL", "local-model")
    SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost:8888")

    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "6"))
    CODE_TIMEOUT_SEC: int = int(os.getenv("CODE_TIMEOUT_SEC", "15"))
    EXTRACTOR_MAX_CHARS: int = int(os.getenv("EXTRACTOR_MAX_CHARS", "4000"))

    SYSTEM_PROMPT: str = """You are a precise AI assistant with access to tools.
RULES:
1. Use web_search for current/factual information.
2. Use web_extractor to get specific details from a URL found via search.
3. Use code_interpreter for math, data analysis, or deterministic computation.
4. NEVER fabricate URLs, dates, or numerical results.
5. When you have sufficient information, respond directly WITHOUT tool calls.
6. If a tool fails, adapt your approach or explain the limitation."""
