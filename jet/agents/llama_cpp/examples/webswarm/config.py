# === LOCAL SERVER ENDPOINTS ===
LLM_BASE_URL = "http://localhost:8080/v1"
EMBEDDER_URL = "http://localhost:8081/embed"
RERANKER_URL = "http://localhost:8082/rerank"
LLM_MODEL_NAME = "local-model"  # Placeholder; llama.cpp ignores this

# === CONTEXT BUDGETS (TOKENS) - TUNE TO YOUR MODEL'S EFFECTIVE CONTEXT ===
BUDGETS = {
    "planner": {"system": 400, "history": 1000, "output": 500},
    "searcher": {"system": 300, "parent_ctx": 300, "docs": 2500, "output": 200},
    "synthesizer": {
        "system": 400,
        "global_index": 1500,
        "top_findings": 3000,
        "output": 1000,
    },
    "compressor": {"system": 200, "input": 2000, "output": 300},
}

# === RECURSION GUARDS ===
MAX_DEPTH = 4
MAX_ITERATIONS = 20
MAX_WALL_SECONDS = 300
MAX_TOTAL_TOKENS = 60000
SEMANTIC_DEDUP_THRESHOLD = 0.85
RERANK_TOP_K = 3
DOC_CHAR_LIMIT = 4000  # Per-document truncation before embedding

# === PATHS ===
GRAMMAR_DIR = "./grammars"
CACHE_DB = "./swarm_cache.db"
VECTOR_DB_PATH = "./chroma_swarm"
