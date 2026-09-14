import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import HTTPSpanExporter, register
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console(force_terminal=True, highlight=False)

_provider: Optional[TracerProvider] = None
_initialized_project: Optional[str] = None


def init_tracing(
    project_name: str,
    phoenix_rest_api: str,
    auto_instrument: bool = False,
) -> TracerProvider:
    """
    Centralized Phoenix OTEL setup using HTTP/Protobuf + BatchSpanProcessor.
    Call once at app startup. Returns the configured TracerProvider.
    """
    global _provider, _initialized_project

    # Idempotency check
    if _provider is not None:
        if _initialized_project == project_name:
            logger.debug(f"Tracing already active for project: {project_name}")
            return _provider
        else:
            logger.warning(
                f"Tracing already initialized for '{_initialized_project}'. "
                f"Ignoring request for '{project_name}'."
            )
            return _provider

    # Normalize endpoint
    base_url = phoenix_rest_api.rstrip("/")
    if base_url.endswith("/v1"):
        traces_endpoint = f"{base_url}/traces"
    elif base_url.endswith("/v1/traces"):
        traces_endpoint = base_url
    else:
        traces_endpoint = f"{base_url}/v1/traces"

    resource = Resource.create({"openinference.project.name": project_name})
    _provider = TracerProvider(resource=resource)

    # Explicitly use HTTP exporter + BatchSpanProcessor
    exporter = HTTPSpanExporter(endpoint=traces_endpoint)
    _provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set global BEFORE register() to prevent Phoenix from creating a default SimpleSpanProcessor
    trace.set_tracer_provider(_provider)

    # Register with Phoenix SDK but DO NOT let it override the global provider
    # We pass endpoint explicitly to ensure consistency
    register(
        project_name=project_name,
        endpoint=traces_endpoint,
        batch=True,  # Hint to Phoenix SDK, though we already set it up manually
        set_global_tracer_provider=False,
    )

    _initialized_project = project_name
    logger.info(
        f"✅ Phoenix OTEL tracing initialized (HTTP/Batch) for project: {project_name}"
    )
    console.print(f"🔭 Tracing active → [link={base_url}]{base_url}[/link]")
    return _provider


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer from the global provider."""
    return trace.get_tracer(name)
