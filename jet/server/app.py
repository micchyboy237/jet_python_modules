from fastapi import FastAPI
from routes import router, load_model
from middlewares import log_exceptions_middleware
from jet.logger import logger

# Enable parallelism for faster LLM tokenizer encoding
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = FastAPI()

app.middleware("http")(log_exceptions_middleware)
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting server...")
    try:
        # Default model
        load_model("tomaarsen/span-marker-mbert-base-multinerd")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
