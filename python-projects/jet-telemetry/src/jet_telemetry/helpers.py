"""
Jet Telemetry: Helper utilities for tracing and observability.
"""

import hashlib

from opentelemetry import trace as otel_trace

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
    """
    Generates a shareable Phoenix trace URL for the current active span.

    Args:
        phoenix_base_url: The root URL of the Phoenix instance.

    Returns:
        A formatted URL string or None if no active span is found.
    """
    current_span = otel_trace.get_current_span()
    if not current_span.is_recording():
        return None

    trace_id = current_span.get_span_context().trace_id
    # Format as 32-char lowercase hex
    trace_id_hex = format(trace_id, "032x")

    base = phoenix_base_url.rstrip("/")
    return f"{base}/redirects/traces/{trace_id_hex}"
