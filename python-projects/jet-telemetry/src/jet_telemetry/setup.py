"""
Summary: Centralized OpenTelemetry setup for Jet's projects.
Uses arize-phoenix-otel to configure tracing with Phoenix-aware defaults.
Reads LLM_OBS_PHOENIX_URL from environment variables for flexible deployment.
"""

import os

import phoenix.otel as pxtl


def initialize_telemetry(service_name: str = "default-service") -> None:
    """
    Initializes the global TracerProvider for the application.

    Args:
        service_name: The name of the microservice or app for identification in Phoenix.
    """
    # Use the team-standard env var with a local development fallback
    phoenix_endpoint = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")

    # Set the environment variable expected by phoenix.otel before registering
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

    # Register Phoenix OTEL with the service name
    # This automatically discovers installed OpenInference instrumentors
    pxtl.register(service_name=service_name)

    print(f"[JetTelemetry] Initialized for '{service_name}' -> {phoenix_endpoint}")
