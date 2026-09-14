import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from phoenix.otel import BatchSpanProcessor, HTTPSpanExporter, TracerProvider, register
from rich.console import Console

logger = logging.getLogger(__name__)

# Module-level singleton console instance
console = Console(force_terminal=True, highlight=False)

_provider: Optional[TracerProvider] = None


def init_tracing(
    project_name: str,
    phoenix_rest_api: str,
    auto_instrument: bool = False,
) -> TracerProvider:
    """
    Centralized Phoenix OTEL setup. Call once at app startup.
    Returns the configured TracerProvider [[6]].
    """
    global _provider
    if _provider is not None:
        logger.warning("Tracing already initialized; returning existing provider")
        return _provider

    resource = Resource.create({"openinference.project.name": project_name})
    _provider = TracerProvider(resource=resource)

    exporter = HTTPSpanExporter(endpoint=f"{phoenix_rest_api.rstrip('/')}/traces")
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)

    # Register with set_global_tracer_provider=False to avoid duplicate processors
    register(project_name=project_name, set_global_tracer_provider=False)

    logger.info(f"Phoenix OTEL tracing initialized for project: {project_name}")
    return _provider


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer from the global provider."""
    return trace.get_tracer(name)
