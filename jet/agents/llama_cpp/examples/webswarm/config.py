import os

# Resolve paths relative to this config file's location
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEARXNG_QUERY_URL = os.getenv("SEARXNG_URL", "http://localhost:8888/search")
SEARXNG_ENGINES = []
SEARXNG_CATEGORIES = ["general"]
SEARXNG_MIN_SCORE = 0.1
SEARXNG_MAX_RESULTS = 10
SEARXNG_USE_CACHE = True

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

MAX_DEPTH = 4
MAX_ITERATIONS = 20
MAX_WALL_SECONDS = 300
MAX_TOTAL_TOKENS = 16000
SEMANTIC_DEDUP_THRESHOLD = 0.85
RERANK_TOP_K = 3
DOC_CHAR_LIMIT = 4000

GRAMMAR_DIR = os.path.join(_THIS_DIR, "grammars")
CACHE_DB = os.path.join(_THIS_DIR, "swarm_cache.db")
VECTOR_DB_PATH = os.path.join(_THIS_DIR, "chroma_swarm")
