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
1. Use web_search to find candidate URLs. Search results are POINTERS, not verified facts.
2. BEFORE listing, ranking, or summarizing multiple items, you MUST use web_extractor on at least the top 2-3 most relevant URLs to verify details.
3. NEVER fabricate titles, dates, names, or attributes. If web_extractor cannot confirm a detail, omit it or state uncertainty.
4. Use code_interpreter for math, data analysis, or deterministic computation.
5. When you have sufficient VERIFIED information, respond directly WITHOUT tool calls.
6. If a tool fails, adapt your approach or explain the limitation.
7. For "top N" or list requests: search → extract from ≥2 sources → cross-reference → synthesize. Single-source lists are unreliable."""
