from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from jet.db.postgres.cleanup import drop_table_if_exists, drop_type_if_exists
from jet.db.postgres.config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from mem0 import Memory

# ──────────────────────────────────────────────
#               CONFIGURATION
# ──────────────────────────────────────────────


def create_memory(collection_name: str = "memories", reset: bool = False) -> Memory:
    if reset:
        drop_table_if_exists(f"public.{collection_name}_entities")
        drop_type_if_exists(f"public.{collection_name}_entities")

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": 12000,
                "openai_base_url": LLM_BASE_URL,
                "api_key": "dummy",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBED_MODEL,
                "embedding_dims": EMBED_DIMS,  # fallback 768 if model not in dict
                "openai_base_url": EMBED_BASE_URL,
                "api_key": "dummy",
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": EMBED_DIMS,
                "dbname": DEFAULT_DB,
                "user": DEFAULT_USER,
                "password": DEFAULT_PASSWORD,
                "host": DEFAULT_HOST,
                "port": DEFAULT_PORT,
            },
        },
    }

    memory = Memory.from_config(config)
    return memory


if __name__ == "__main__":
    collection_name = "default_memory"
    memory = create_memory(collection_name)
    # Example: Add a memory (can be a string or list of messages)
    memory.add("I'm visiting Paris next month and love French cuisine.", user_id="john")

    # Or add from conversation messages
    messages = [
        {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
        {"role": "assistant", "content": "Hey Alex! I'll remember that."},
    ]
    memory.add(messages, user_id="alex")

    # Retrieve memories
    memories = memory.get_all(user_id="john")
    for mem in memories:
        print(mem)

    # Search relevant memories
    results = memory.search("What are my travel plans?", user_id="john")
    print(results)
