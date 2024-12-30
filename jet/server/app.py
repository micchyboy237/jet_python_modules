from fastapi import FastAPI
from routes.rag import router as rag_router
from routes.ner import router as ner_router
from middlewares import log_exceptions_middleware
from jet.logger import logger

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = FastAPI()

app.middleware("http")(log_exceptions_middleware)

# Include the routes
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(ner_router, prefix="/api/v1/ner", tags=["NER"])
