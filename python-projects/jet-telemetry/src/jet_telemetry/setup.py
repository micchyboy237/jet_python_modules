"""
Summary: Centralized OpenTelemetry setup for Jet's projects.
Uses arize-phoenix-otel to configure tracing with Phoenix-aware defaults.
"""

import os

import phoenix.otel as pxtl


def initialize_telemetry(
    service_name: str = "default-service", endpoint: str | None = None
) -> None:
    """
    Initializes the global TracerProvider for the application.

    Args:
        service_name: The name of the microservice or app for identification in Phoenix.
        endpoint: Optional override for the Phoenix collector URL.
    """
    # Priority: 1. Argument, 2. Env Var, 3. Local Default
    phoenix_endpoint = endpoint or os.getenv(
        "PHOENIX_ENDPOINT", "http://localhost:6006"
    )

    # Set the environment variable expected by phoenix.otel before registering
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

    # Register Phoenix OTEL with the service name
    pxtl.register(service_name=service_name)

    print(f"[JetTelemetry] Initialized for '{service_name}' -> {phoenix_endpoint}")
