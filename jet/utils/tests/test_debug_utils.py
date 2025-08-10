import pytest
from typing import Dict, Any
from jet.utils.debug_utils import get_non_function_locals


@pytest.fixture
def setup_locals():
    """Fixture to clean up any shared state if needed."""
    yield
    # No specific cleanup required for this case


class TestGetNonFunctionLocals:
    def test_retrieves_only_non_function_variables(self, setup_locals):
        # Given: A local scope with mixed variables (non-functions and a function)
        x = 10
        y = "test"

        def dummy_func():
            pass
        z = [1, 2, 3]
        expected = {"x": 10, "y": "test", "z": [1, 2, 3]}

        # When: We call get_non_function_locals
        result = get_non_function_locals()

        # Then: Only non-function variables are returned
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_empty_locals(self, setup_locals):
        # Given: An empty local scope
        expected: Dict[str, Any] = {}

        # When: We call get_non_function_locals
        result = get_non_function_locals()

        # Then: An empty dictionary is returned
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_only_functions(self, setup_locals):
        # Given: A local scope with only a function
        def another_func():
            pass
        expected: Dict[str, Any] = {}

        # When: We call get_non_function_locals
        result = get_non_function_locals()

        # Then: An empty dictionary is returned
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_non_function_callables(self, setup_locals):
        # Given: A local scope with a class (callable) and a non-callable
        class MyClass:
            pass
        value = 100
        expected = {"value": 100}

        # When: We call get_non_function_locals
        result = get_non_function_locals()

        # Then: Only non-callable variables are returned
        assert result == expected, f"Expected {expected}, but got {result}"
