from .routes.ner import predict_entities
from .helpers.rag import RAG

import nest_asyncio

nest_asyncio.apply()

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
