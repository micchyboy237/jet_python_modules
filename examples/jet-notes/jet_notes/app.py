import logging
import os
import sys
from contextlib import asynccontextmanager

# --- Startup Configuration (Must happen before local imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Add parent directory to sys.path so 'jet_notes' package is resolvable
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load environment variables before any telemetry or app initialization
try:
    from dotenv import load_dotenv

    env_file = ".env.development" if os.path.exists(".env.development") else ".env"
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed

# Configure logging early for startup visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Local Imports (Now safe because sys.path is updated) ---
from fastapi import FastAPI
from jet_notes.telemetry import initialize_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan event handler.
    Handles startup and shutdown logic.
    """
    logger.info("Starting up jet_notes API...")
    try:
        initialize_tracing(
            project_name="jet_notes-api",
            auto_instrument=True,
            batch=True,
        )
        logger.info("Telemetry initialized for API.")
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
    yield
    logger.info("Shutting down jet_notes API...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/notes")
async def create_note_endpoint(title: str, content: str):
    from jet_notes.services.note_service import NoteService

    service = NoteService()
    note = service.create_note(title, content)
    return {"id": note.id, "title": note.title}


if __name__ == "__main__":
    # Dynamic server configuration via environment variables
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "true").lower() == "true"

    import uvicorn

    logger.info(f"Starting development server at http://{host}:{port}")
    uvicorn.run(
        "jet_notes.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level="info",
    )
