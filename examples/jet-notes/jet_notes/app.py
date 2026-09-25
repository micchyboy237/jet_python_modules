"""
FastAPI Application Entry Point.
CRITICAL: Telemetry must be initialized BEFORE importing FastAPI or any instrumented libraries.
"""

import logging
import os
import sys

# --- Path Setup (Temporary until proper packaging is used) ---
# Set cwd to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the parent of this file's parent directory to sys.path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Env Loading ---
try:
    from dotenv import load_dotenv

    env_file = ".env.development" if os.path.exists(".env.development") else ".env"
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- CRITICAL: Initialize Tracing FIRST ---
from jet_notes.telemetry import initialize_tracing

try:
    initialize_tracing(
        project_name="jet_notes-api",
        auto_instrument=True,
        batch=True,
    )
    logger.info("Telemetry initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize telemetry: {e}", exc_info=True)
    # Depending on requirements, you might want to exit here if tracing is mandatory

# --- NOW Safe to Import Instrumented Libraries ---
from contextlib import asynccontextmanager

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up jet_notes API...")
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
