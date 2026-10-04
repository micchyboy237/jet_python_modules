"""
Jet Telemetry: Helper utilities for tracing and observability.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from opentelemetry import trace as otel_trace

try:
    from phoenix.client import Client

    PHOENIX_CLIENT_AVAILABLE = True
except ImportError:
    PHOENIX_CLIENT_AVAILABLE = False

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def get_service_name() -> str | None:
    """
    Retrieves the current service name from environment or tracer provider.
    Returns:
        The service name, or None if not initialized.
    """
    service_name = os.getenv("PHOENIX_PROJECT_NAME")
    if service_name:
        return service_name

    try:
        from .setup import get_tracer_provider

        provider = get_tracer_provider()
        if provider:
            return provider.resource.attributes.get("service.name")
    except Exception:
        pass
    return None


def get_project_name() -> str | None:
    """
    Retrieves the current Phoenix Project Name.

    In Jet Telemetry, the Phoenix Project Name is mapped from the
    OpenTelemetry 'service.name' resource attribute.

    Returns:
        The project name, or None if telemetry is not initialized.
    """
    return get_service_name()


def _derive_phoenix_base_url() -> str:
    """
    Determines the Phoenix Base URL by checking:
    1. Explicit env var LLM_OBS_PHOENIX_URL
    2. Collector endpoint env var PHOENIX_COLLECTOR_ENDPOINT (stripping /v1/traces)
    3. Default localhost
    """
    # 1. Check specific UI URL env var
    ui_url = os.getenv("LLM_OBS_PHOENIX_URL")
    if ui_url:
        return ui_url.rstrip("/")

    # 2. Derive from Collector Endpoint
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if collector_endpoint:
        base = collector_endpoint
        # Strip common API suffixes to get the UI root
        if base.endswith("/v1/traces"):
            base = base[: -len("/v1/traces")]
        elif base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/")

    # 3. Fallback
    return "http://localhost:6006"


def redact(text: str) -> str:
    """Redact sensitive content from text before tracing."""
    lower = text.lower()
    for pattern in PII_PATTERNS:
        if pattern in lower:
            return "[REDACTED: contains sensitive content]"
    return text


def hash_prompt(prompt: str) -> str:
    """Create a short deterministic hash of a prompt for version tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def get_trace_url(phoenix_base_url: str | None = None) -> str | None:
    """
    Generates a shareable Phoenix trace URL for the current active span.

    Args:
        phoenix_base_url: Optional override. If not provided, it automatically
                          detects the URL from PHOENIX_COLLECTOR_ENDPOINT or defaults
                          to http://localhost:6006.
    """
    current_span = otel_trace.get_current_span()
    if not current_span.is_recording():
        return None

    trace_id = current_span.get_span_context().trace_id
    trace_id_hex = format(trace_id, "032x")

    # Use provided URL, or auto-detect from config/env
    base = phoenix_base_url if phoenix_base_url else _derive_phoenix_base_url()
    base = base.rstrip("/")

    return f"{base}/redirects/traces/{trace_id_hex}"


def get_spans_api_url(
    phoenix_base_url: str | None = None,
    project_name: str | None = None,
    trace_id: str | None = None,
    limit: int = 1000,
) -> str | None:
    """
    Generates a Phoenix REST API URL for manual inspection.

    Args:
        phoenix_base_url: Optional override for the Phoenix UI base URL.
        project_name: Optional project name. Defaults to current active project.
        trace_id: Optional trace ID to filter spans.
        limit: Maximum number of spans to return.

    Returns:
        The formatted API URL, or None if project name cannot be resolved.
    """
    if project_name is None:
        project_name = get_project_name()

    if not project_name:
        print("⚠️ Cannot generate spans API URL: No project name available.")
        return None

    base = (phoenix_base_url or _derive_phoenix_base_url()).rstrip("/")
    url = f"{base}/v1/projects/{project_name}/spans?limit={limit}"
    if trace_id:
        url += f"&trace_id={trace_id}"
    return url


def export_spans_to_jsonl(
    project_name: str | None = None,
    trace_id: str | None = None,
    output_path: str | Path = "spans.jsonl",
    phoenix_base_url: str | None = None,
    limit: int = 1000,
    wait_for_flush: bool = True,
) -> Path:
    """
    Export spans as RAW JSON objects (preserving nesting, events, and all attributes).
    Performs a single fetch attempt after flushing traces.

    Args:
        project_name: Optional project name. Defaults to current active project.
        trace_id: Optional trace ID to filter by. Exports all recent spans if None.
        output_path: Destination file path for the JSONL export.
        phoenix_base_url: Optional override for the Phoenix UI base URL.
        limit: Maximum number of spans to fetch.
        wait_for_flush: Whether to force flush traces before exporting.
    """
    if not PHOENIX_CLIENT_AVAILABLE:
        raise ImportError("arize-phoenix-client is required for span export.")

    # Resolve project name if not provided
    resolved_project = project_name or get_project_name()
    if not resolved_project:
        print("❌ Export failed: No project name provided and no active project found.")
        output_path = Path(output_path)
        output_path.write_text("")
        return output_path

    from datetime import datetime, timedelta

    # Auto-detect base URL if not provided
    if phoenix_base_url is None:
        phoenix_base_url = _derive_phoenix_base_url()

    if wait_for_flush:
        force_flush_traces()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = Client(base_url=phoenix_base_url)

    try:
        trace_ids = [trace_id] if trace_id else None
        spans_list = client.spans.get_spans(
            project_identifier=resolved_project,
            trace_ids=trace_ids,
            limit=limit,
            start_time=datetime.now() - timedelta(days=7),
        )

        if not spans_list:
            print(
                f"⚠️ No spans found for project '{resolved_project}' (trace: {trace_id})."
            )
            output_path.write_text("")
            return output_path

        spans_list.sort(key=lambda s: s.get("start_time", ""))

        with open(output_path, "w") as f:
            for span in spans_list:
                f.write(json.dumps(span) + "\n")

        print(f"✅ Exported {len(spans_list)} spans to {output_path.name}")
        return output_path

    except Exception as e:
        print(f"❌ Export failed: {e}")
        output_path.write_text("")
        return output_path


def force_flush_traces(timeout_ms: int = 5000):
    """Forces the OpenTelemetry tracer provider to flush all pending spans."""
    try:
        from .setup import get_tracer_provider

        provider = get_tracer_provider()
        if provider:
            success = provider.force_flush(timeout_millis=timeout_ms)
            if not success:
                print("⚠️ Tracer provider flush timed out or failed.")
            else:
                time.sleep(1.5)
    except Exception as e:
        print(f"⚠️ Failed to force flush traces: {e}")
