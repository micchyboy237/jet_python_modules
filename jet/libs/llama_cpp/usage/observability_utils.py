import atexit
import logging
import os
import threading

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from phoenix.otel import register
from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")

# --- idempotent observability (once per process) ---
_OBS_LOCK = threading.Lock()
_OBS_INITIALIZED = False
_OBS_PROJECT: str | None = None
_OBS_TRACER_PROVIDER = None


def setup_observability(
    project_name: str = "chat-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
):
    """
    Register Phoenix + OpenAI instrumentor once per process.

    Later calls are no-ops (avoids "already instrumented" and
    "Overriding of current TracerProvider is not allowed").
    """
    global _OBS_INITIALIZED, _OBS_PROJECT, _OBS_TRACER_PROVIDER

    with _OBS_LOCK:
        if _OBS_INITIALIZED:
            if project_name != _OBS_PROJECT:
                logger.debug(
                    "Observability already initialized for project=%s; "
                    "ignoring request for project=%s",
                    _OBS_PROJECT,
                    project_name,
                )
            else:
                logger.debug(
                    "Observability already initialized for project=%s; skipping",
                    project_name,
                )
            return _OBS_TRACER_PROVIDER

        if capture_content:
            os.environ.setdefault(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
                "SPAN_AND_EVENT",
            )

        # batch=True → BatchSpanProcessor with OTel defaults:
        #   schedule_delay_millis=5000, max_queue_size=2048,
        #   max_export_batch_size=512, export_timeout_millis=30000
        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{phoenix_url}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
        )
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        _OBS_TRACER_PROVIDER = tracer_provider
        _OBS_PROJECT = project_name
        _OBS_INITIALIZED = True

        # REQUIRED when batch=True: flush buffered spans on exit.
        # Without this, short-lived scripts lose the final batch silently
        # because the 5s flush timer may never fire before process termination.
        atexit.register(tracer_provider.shutdown)

        console.print(
            f"🔭 Observability enabled → [link={phoenix_url}]{phoenix_url}[/link]"
        )
        logger.info(f"📁 Phoenix project name: {project_name}")
        return tracer_provider
