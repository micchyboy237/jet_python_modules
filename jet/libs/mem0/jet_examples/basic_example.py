from mem0 import Memory

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test_memories",  # or your own name
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,  # Match your embedder's dimensions
        },
    },
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.1:latest",  # Your local chat model
            "temperature": 0.0,
            "max_tokens": 2000,
            "ollama_base_url": "http://localhost:11434",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text:latest",  # Local embedding model
            "ollama_base_url": "http://localhost:11434",
        },
    },
    # Optional: history store (defaults to local SQLite)
    # "history": {"provider": "sqlite", "config": {"path": "~/.mem0/history.db"}}
}

# Initialize Mem0 with the local config
m = Memory.from_config(config)

# Example: Add a memory (can be a string or list of messages)
m.add("I'm visiting Paris next month and love French cuisine.", user_id="john")

# Or add from conversation messages
messages = [
    {"role": "user", "content": "Hi, I'm Alex. I love basketball and gaming."},
    {"role": "assistant", "content": "Hey Alex! I'll remember that."},
]
m.add(messages, user_id="alex")

# Retrieve memories
memories = m.get_all(user_id="john")
for mem in memories:
    print(mem)

# Search relevant memories
results = m.search("What are my travel plans?", user_id="john")
print(results)
