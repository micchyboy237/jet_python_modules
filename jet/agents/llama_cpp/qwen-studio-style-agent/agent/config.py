import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    LLAMA_BASE_URL: str = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
    LLAMA_MODEL: str = os.getenv("LLAMA_CPP_MODEL", "qwen2.5-7b-instruct-q4_k_m")
    SEARXNG_URL: str = os.getenv("SEARXNG_URL", "http://localhost:8888")
    PHOENIX_URL: str = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")

    MAX_OUTER_ITERATIONS: int = int(os.getenv("MAX_OUTER_ITERATIONS", "6"))
    INNER_TOOL_ROUNDS: int = int(os.getenv("INNER_TOOL_ROUNDS", "3"))
    CODE_TIMEOUT_SEC: int = int(os.getenv("CODE_TIMEOUT_SEC", "15"))
    EXTRACTOR_MAX_CHARS: int = int(os.getenv("EXTRACTOR_MAX_CHARS", "4000"))

    SYSTEM_PROMPT: str = """You are a precise AI assistant replicating the Qwen Studio workflow.
RULES:
1. Use web_search for current/factual information.
2. Use web_extractor to get specific details from a URL found via search.
3. Use code_interpreter for math, data analysis, or deterministic computation.
4. NEVER fabricate URLs, dates, or numerical results.
5. When you have sufficient information, respond directly WITHOUT tool calls.
6. If a tool fails, adapt your approach or explain the limitation."""
