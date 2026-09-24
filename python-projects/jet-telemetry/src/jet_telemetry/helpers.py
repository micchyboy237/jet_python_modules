# jet_python_modules/python-projects/jet-telemetry/src/jet_telemetry/helpers.py
"""
jet_python_modules/python-projects/jet-telemetry/src/jet_telemetry/helpers.py
Updated for: arize-phoenix-client==3.3.0, arize-phoenix==20.4.0
"""

import hashlib
import time
from pathlib import Path

from opentelemetry import trace as otel_trace

try:
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

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
    """
    Generates a Phoenix REST API URL for manual inspection.
    Note: This is a GET endpoint for browser/curl convenience,
    but the SDK uses POST /v1/spans internally.
    """
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
                # Allow network transmission to complete
                time.sleep(1.0)
    except Exception as e:
        print(f"⚠️ Failed to force flush traces: {e}")


def export_spans_to_jsonl(
    project_name: str,
    trace_id: str | None = None,
    output_path: str | Path = "spans.jsonl",
    phoenix_base_url: str = "http://localhost:6006",
    limit: int = 1000,
    wait_for_flush: bool = True,
    max_retries: int = 3,
) -> Path:
    """
    Export spans from Phoenix to JSONL using arize-phoenix-client SDK.
    Handles race conditions with BatchSpanProcessor via retries.
    """
    if not PHOENIX_CLIENT_AVAILABLE:
        raise ImportError(
            "arize-phoenix-client is required. Install with: pip install arize-phoenix-client"
        )

    from datetime import datetime, timedelta

    if wait_for_flush:
        force_flush_traces()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = Client(base_url=phoenix_base_url)

    # Build query using SpanQuery DSL (required for v3.3.0+)
    query = SpanQuery()
    if trace_id:
        # Use double equals for SpanQuery filter syntax
        query = query.where(f"trace_id == '{trace_id}'")

    last_error = None
    for attempt in range(max_retries):
        try:
            # Use project_identifier (preferred over project_name in v3.3.0)
            spans_df = client.spans.get_spans_dataframe(
                project_identifier=project_name,
                query=query,
                limit=limit,
                start_time=datetime.now() - timedelta(days=7),
            )

            if spans_df.empty:
                if attempt < max_retries - 1:
                    print(
                        f"⏳ No spans found yet (attempt {attempt + 1}/{max_retries}), retrying..."
                    )
                    time.sleep(2.0)
                    continue
                print(
                    f"⚠️ No spans found for project='{project_name}', trace_id='{trace_id}'"
                )
                output_path.write_text("")
                return output_path

            spans_df.to_json(str(output_path), orient="records", lines=True)
            print(f"✅ Exported {len(spans_df)} spans to {output_path}")
            return output_path

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(
                    f"⏳ Export failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying..."
                )
                time.sleep(2.0)
            else:
                print(f"❌ Failed to export spans after {max_retries} attempts: {e}")
                # Create empty file to indicate failure/completion
                output_path.write_text("")
                return output_path

    return output_path


def export_spans_to_csv(
    project_name: str,
    trace_id: str | None = None,
    output_path: str | Path = "spans.csv",
    phoenix_base_url: str = "http://localhost:6006",
    limit: int = 1000,
    wait_for_flush: bool = True,
) -> Path:
    """
    Export spans from Phoenix to CSV format using arize-phoenix-client SDK.
    Args:
        project_name: Name of the Phoenix project.
        trace_id: Optional trace ID to filter spans by.
        output_path: File path to save CSV output.
        phoenix_base_url: Base URL of the Phoenix instance.
        limit: Maximum number of spans to retrieve.
        wait_for_flush: If True, forces a flush of pending spans before exporting.
    Returns:
        Path to the exported CSV file.
    """
    if not PHOENIX_CLIENT_AVAILABLE:
        raise ImportError(
            "arize-phoenix-client is required. Install with: pip install arize-phoenix-client"
        )

    from datetime import datetime, timedelta

    if wait_for_flush:
        force_flush_traces()

    try:
        client = Client(base_url=phoenix_base_url)

        query = None
        if trace_id:
            query = SpanQuery().where(f"trace_id = '{trace_id}'")

        spans_df = client.spans.get_spans_dataframe(
            project_identifier=project_name,
            limit=limit,
            query=query,
            start_time=datetime.now() - timedelta(days=7),
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if spans_df.empty:
            print(
                f"⚠️ No spans found for project='{project_name}', trace_id='{trace_id}'"
            )
            output_path.write_text("")
            return output_path

        spans_df.to_csv(str(output_path), index=False)
        print(f"✅ Exported {len(spans_df)} spans to {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Failed to export spans: {e}")
        return Path(output_path)
