"""
Telemetry initialization module for jet_notes.
Follows OpenTelemetry industry standard pattern:
- Initialize once at application startup
- Configure global TracerProvider before any instrumented code runs
- Use environment variables for configuration
"""

import logging
import os
from typing import Optional

from opentelemetry.sdk.trace import TracerProvider
from phoenix.otel import register

logger = logging.getLogger(__name__)

# Flag to prevent double initialization
_is_initialized = False


def initialize_tracing(
    project_name: Optional[str] = None,
    auto_instrument: bool = True,
    batch: bool = True,
    endpoint: Optional[str] = None,
    protocol: str = "http/protobuf",
) -> TracerProvider:
    """
    Initialize OpenTelemetry tracing with Phoenix integration.

    This should be called ONCE at application startup, before any
    instrumented libraries are imported or used.

    Args:
        project_name: Name of the project in Phoenix. Defaults to env var or 'jet_notes'.
        auto_instrument: Enable automatic instrumentation of AI/ML libraries.
        batch: Enable batch span processing for production performance.
        endpoint: Custom Phoenix collector endpoint. Defaults to env var.
        protocol: Transport protocol ('http/protobuf' or 'grpc').

    Returns:
        Configured TracerProvider instance
    """
    global _is_initialized

    if _is_initialized:
        logger.warning("Tracing has already been initialized. Skipping.")
        # Return a dummy provider or the current global one if accessible
        from opentelemetry import trace

        return trace.get_tracer_provider()

    # Load defaults from environment if not provided
    if project_name is None:
        project_name = os.getenv("PHOENIX_PROJECT_NAME", "jet_notes")

    if endpoint is None:
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

    api_key = os.getenv("PHOENIX_API_KEY")

    logger.info(f"Initializing Phoenix OTEL for project: {project_name}")
    logger.info(f"Endpoint: {endpoint or 'Default (localhost)'}")
    logger.info(f"Protocol: {protocol}")

    try:
        kwargs = {
            "project_name": project_name,
            "auto_instrument": auto_instrument,
            "batch": batch,
            "protocol": protocol,
        }

        if endpoint:
            kwargs["endpoint"] = endpoint

        if api_key:
            kwargs["api_key"] = api_key

        tracer_provider = register(**kwargs)
        _is_initialized = True
        logger.info("Phoenix OTEL tracing initialized successfully.")
        return tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize Phoenix OTEL tracing: {e}", exc_info=True)
        # In production, you might want to raise this or fail fast depending on requirements
        # For robustness, we re-raise to alert the developer/operator
        raise
