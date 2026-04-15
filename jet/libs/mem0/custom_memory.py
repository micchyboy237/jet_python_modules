import os

from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_EMBEDDING_SIZES
from mem0 import Memory

# ──────────────────────────────────────────────
#               CONFIGURATION
# ──────────────────────────────────────────────


def create_memory(collection_name: str = "memories") -> Memory:
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLAMA_CPP_LLM_MODEL"),
                "temperature": 0.7,
                "max_tokens": 12000,
                "openai_base_url": os.getenv("LLAMA_CPP_LLM_URL"),
                "api_key": "dummy",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("LLAMA_CPP_EMBED_MODEL"),
                "embedding_dims": LLAMACPP_MODEL_EMBEDDING_SIZES.get(
                    os.getenv("LLAMA_CPP_EMBED_MODEL"), 768
                ),  # fallback 768 if model not in dict
                "openai_base_url": os.getenv("LLAMA_CPP_EMBED_URL"),
                "api_key": "dummy",
            },
        },
        # "vector_store": {
        #     "provider": "chroma",
        #     "config": {
        #         "collection_name": collection_name,
        #         "path": str(
        #             Path("~/.cache/chroma_db/basic_mem0_chatbot").expanduser().resolve()
        #         ),
        #     },
        # },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": LLAMACPP_MODEL_EMBEDDING_SIZES.get(
                    os.getenv("LLAMA_CPP_EMBED_MODEL"), 768
                ),
                "user": os.getenv("DB_POSTGRES_USER", "test"),
                "password": os.getenv("DB_POSTGRES_PASSWORD", "123"),
                "host": os.getenv("DB_POSTGRES_HOST", "127.0.0.1"),
                "port": int(os.getenv("DB_POSTGRES_PORT", 5432)),
                "dbname": os.getenv(
                    "DB_VECTOR_DBNAME", "postgres"
                ),  # Changed to default 'postgres' db
                "diskann": False,  # Optional, requires pgvectorscale extension
                "hnsw": False,  # Optional, for HNSW indexing
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
