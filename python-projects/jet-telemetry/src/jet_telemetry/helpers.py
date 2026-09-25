# jet_python_modules/python-projects/jet-telemetry/src/jet_telemetry/helpers.py
"""
Jet Telemetry: Helper utilities for tracing and observability.
"""

import hashlib
import json
import time
from pathlib import Path

from opentelemetry import trace as otel_trace

try:
    from phoenix.client import Client

    PHOENIX_CLIENT_AVAILABLE = True
except ImportError:
    PHOENIX_CLIENT_AVAILABLE = False

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


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


def get_trace_url(phoenix_base_url: str = "http://localhost:6006") -> str | None:
    """Generates a shareable Phoenix trace URL for the current active span."""
    current_span = otel_trace.get_current_span()
    if not current_span.is_recording():
        return None

    trace_id = current_span.get_span_context().trace_id
    trace_id_hex = format(trace_id, "032x")

    base = phoenix_base_url.rstrip("/")
    return f"{base}/redirects/traces/{trace_id_hex}"


def get_spans_api_url(
    phoenix_base_url: str,
    project_name: str,
    trace_id: str | None = None,
    limit: int = 1000,
) -> str:
    """Generates a Phoenix REST API URL for manual inspection."""
    base = phoenix_base_url.rstrip("/")
    url = f"{base}/v1/projects/{project_name}/spans?limit={limit}"
    if trace_id:
        url += f"&trace_id={trace_id}"
    return url


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
                # Fixed wait to allow server-side indexing
                time.sleep(1.5)
    except Exception as e:
        print(f"⚠️ Failed to force flush traces: {e}")


def export_spans_to_jsonl(
    project_name: str,
    trace_id: str | None = None,
    output_path: str | Path = "spans.jsonl",
    phoenix_base_url: str = "http://localhost:6006",
    limit: int = 1000,
    wait_for_flush: bool = True,
) -> Path:
    """
    Export spans as RAW JSON objects (preserving nesting, events, and all attributes).
    Performs a single fetch attempt after flushing traces.
    """
    if not PHOENIX_CLIENT_AVAILABLE:
        raise ImportError("arize-phoenix-client is required.")

    from datetime import datetime, timedelta

    if wait_for_flush:
        force_flush_traces()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = Client(base_url=phoenix_base_url)

    try:
        trace_ids = [trace_id] if trace_id else None

        spans_list = client.spans.get_spans(
            project_identifier=project_name,
            trace_ids=trace_ids,
            limit=limit,
            start_time=datetime.now() - timedelta(days=7),
        )

        if not spans_list:
            print(f"⚠️ No spans found for trace '{trace_id}'.")
            output_path.write_text("")
            return output_path

        # Sort spans by start_time to maintain hierarchy order
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
