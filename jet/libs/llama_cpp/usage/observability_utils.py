import os
import threading

from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from phoenix.otel import register
from rich.console import Console

console = Console()

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

        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{phoenix_url}/v1/traces",
            batch=False,
            # First (and only) registration owns the global provider
            set_global_tracer_provider=True,
        )
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        _OBS_TRACER_PROVIDER = tracer_provider
        _OBS_PROJECT = project_name
        _OBS_INITIALIZED = True

        console.print(
            f"🔭 Observability enabled → [link={phoenix_url}]{phoenix_url}[/link]"
        )
        logger.info(f"📁 Phoenix project name: {project_name}")
        return tracer_provider
