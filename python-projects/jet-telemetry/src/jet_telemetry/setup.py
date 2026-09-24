"""
Summary: Centralized OpenTelemetry setup for Jet's projects.
Uses arize-phoenix-otel 0.17.1+ with proper HTTP/Protobuf configuration.
"""

import logging
import os

import phoenix.otel as pxtl

logger = logging.getLogger(__name__)
_initialized = False
_tracer_provider = None


def get_tracer_provider():
    """Get the global tracer provider."""
    return _tracer_provider


def initialize_telemetry(
    service_name: str = "default-service",
    endpoint: str | None = None,
    protocol: str | None = None,
    auto_instrument: bool = True,
    batch: bool = True,
) -> None:
    """
    Initializes the global TracerProvider for the application.
    Args:
        service_name: The name of the microservice or app (maps to Phoenix project_name).
        endpoint: Optional override for the Phoenix collector URL.
        protocol: Transport protocol: "http/protobuf" or "grpc".
                  Defaults to "http/protobuf" if endpoint contains port 6006, else "grpc".
        auto_instrument: Enable automatic instrumentation for AI/ML libraries.
        batch: Use batch span processing for production performance.
    """
    global _initialized, _tracer_provider

    if _initialized:
        logger.debug(f"Telemetry already initialized for '{service_name}'. Skipping.")
        return

    base_endpoint = endpoint or os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    )

    if base_endpoint.endswith("/v1"):
        collector_endpoint = f"{base_endpoint}/traces"
    elif not base_endpoint.endswith("/v1/traces"):
        if "6006" in base_endpoint:
            collector_endpoint = f"{base_endpoint}/v1/traces"
        else:
            collector_endpoint = base_endpoint
    else:
        collector_endpoint = base_endpoint

    if protocol is None:
        if "4317" in collector_endpoint:
            protocol = "grpc"
        else:
            protocol = "http/protobuf"

    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = collector_endpoint
    os.environ["PHOENIX_PROJECT_NAME"] = service_name

    try:
        _tracer_provider = pxtl.register(
            project_name=service_name,
            endpoint=collector_endpoint,
            protocol=protocol,
            auto_instrument=auto_instrument,
            batch=batch,
        )
        _initialized = True
        print(
            f"[JetTelemetry] Initialized '{service_name}' -> {collector_endpoint} ({protocol})"
        )
    except Exception as e:
        logger.error(f"Failed to initialize telemetry: {e}")
        raise
