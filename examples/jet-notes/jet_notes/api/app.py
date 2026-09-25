import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from jet_notes.telemetry import initialize_tracing

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan event handler.
    Handles startup and shutdown logic.
    """
    # Startup
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

    # Shutdown
    logger.info("Shutting down jet_notes API...")
    # Add any cleanup logic here if needed


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
