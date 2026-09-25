from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture(autouse=True)
def disable_tracing_for_tests():
    """Disable tracing during tests to avoid overhead and external calls."""
    with patch("phoenix.otel.register") as mock_register:
        # Return a simple no-op or basic provider for tests
        mock_register.return_value = TracerProvider()
        yield
