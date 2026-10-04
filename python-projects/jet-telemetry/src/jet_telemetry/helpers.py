"""
Jet Telemetry: Helper utilities for tracing and observability.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from opentelemetry import trace as otel_trace

# Global storage for the last known trace context (Thread-safe via simple assignment for now)
_last_trace_context = {"trace_id": None, "project_name": None}

try:
    from phoenix.client import Client

    PHOENIX_CLIENT_AVAILABLE = True
except ImportError:
    PHOENIX_CLIENT_AVAILABLE = False

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def _update_last_trace_context(trace_id_hex: str | None):
    """Internal helper to remember the last active trace."""
    if trace_id_hex:
        _last_trace_context["trace_id"] = trace_id_hex
        # Also try to grab the project name while we're at it
        from .setup import get_tracer_provider

        provider = get_tracer_provider()
        if provider:
            _last_trace_context["project_name"] = provider.resource.attributes.get(
                "openinference.project.name"
            )


def get_provider_resource():
    """
    Retrieves the entire Resource object from the current tracer provider.
    """
    from .setup import get_tracer_provider

    provider = get_tracer_provider()
    if provider:
        return provider.resource
    return None


def get_resource(attribute: str) -> str | None:
    """Retrieves a specific resource value from the current tracer provider."""
    resource = get_provider_resource()
    if resource:
        return resource.attributes.get(attribute)
    return None


def get_service_name() -> str | None:
    return get_resource("service.name")


def get_project_name() -> str | None:
    # Prefer live resource, fall back to last known
    live = get_resource("openinference.project.name")
    return live or _last_trace_context.get("project_name")


def get_trace_id() -> str | None:
    """
    Retrieves the current active Trace ID.
    If no span is active, returns the last known Trace ID from this session.
    """
    current_span = otel_trace.get_current_span()
    if current_span.is_recording():
        trace_id = current_span.get_span_context().trace_id
        trace_id_hex = format(trace_id, "032x")
        _update_last_trace_context(trace_id_hex)
        return trace_id_hex

    # Lazy fallback: Return the last captured trace ID
    return _last_trace_context.get("trace_id")


def _derive_phoenix_base_url() -> str:
    ui_url = os.getenv("LLM_OBS_PHOENIX_URL")
    if ui_url:
        return ui_url.rstrip("/")
    collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if collector_endpoint:
        base = collector_endpoint
        if base.endswith("/v1/traces"):
            base = base[: -len("/v1/traces")]
        elif base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base.rstrip("/")
    return "http://localhost:6006"


def redact(text: str) -> str:
    lower = text.lower()
    for pattern in PII_PATTERNS:
        if pattern in lower:
            return "[REDACTED: contains sensitive content]"
    return text


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def get_trace_url(phoenix_base_url: str | None = None) -> str | None:
    """Generates a shareable Phoenix trace URL using the lazy trace ID."""
    trace_id_hex = get_trace_id()
    if not trace_id_hex:
        return None
    base = phoenix_base_url if phoenix_base_url else _derive_phoenix_base_url()
    base = base.rstrip("/")
    return f"{base}/redirects/traces/{trace_id_hex}"


def get_spans_api_url(
    phoenix_base_url: str | None = None,
    project_name: str | None = None,
    trace_id: str | None = None,
    limit: int = 1000,
) -> str | None:
    if project_name is None:
        project_name = get_project_name()
    if not project_name:
        return None

    if trace_id is None:
        trace_id = get_trace_id()

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
    if not PHOENIX_CLIENT_AVAILABLE:
        raise ImportError("arize-phoenix-client is required for span export.")

    resolved_project = project_name or get_project_name()
    if not resolved_project:
        print("❌ Export failed: No project name provided and no active project found.")
        output_path = Path(output_path)
        output_path.write_text("")
        return output_path

    from datetime import datetime, timedelta

    if phoenix_base_url is None:
        phoenix_base_url = _derive_phoenix_base_url()

    if wait_for_flush:
        force_flush_traces()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = Client(base_url=phoenix_base_url)

    try:
        if trace_id is None:
            trace_id = get_trace_id()

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


def display_all_resources():
    resource = get_provider_resource()
    if not resource:
        print("⚠️ No tracer provider initialized. Call initialize_telemetry() first.")
        return

    print("=== Current OpenTelemetry Resource Attributes ===")
    if not resource.attributes:
        print("  (No attributes found)")
    else:
        for key in sorted(resource.attributes.keys()):
            value = resource.attributes[key]
            print(f"  {key}: {value}")
