import os

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model

# LLM model configuration
LLM_MODEL = "gpt-3.5-turbo"  # OpenAI chat model; can switch to "gemini-1.5-flash" etc.

# Document splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval parameters
RETRIEVAL_K = 3  # Number of top chunks to retrieve

# Load API key from environment (do not hardcode)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set this via export or .env