"""
Summary: Centralized OpenTelemetry setup for Jet's projects.
Uses arize-phoenix-otel 0.17.1+ with proper Resource-based service naming.
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

    # Set the environment variables expected by phoenix.otel before registering
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

    # In OTEL 1.44+, service name is set via OTEL_SERVICE_NAME env var
    # This is read automatically by register() when creating the Resource [[29]]
    os.environ["OTEL_SERVICE_NAME"] = service_name

    # Register Phoenix OTEL - do NOT pass service_name directly
    # The SDK reads OTEL_SERVICE_NAME and PHOENIX_COLLECTOR_ENDPOINT from env [[1]]
    pxtl.register()

    print(f"[JetTelemetry] Initialized for '{service_name}' -> {phoenix_endpoint}")
