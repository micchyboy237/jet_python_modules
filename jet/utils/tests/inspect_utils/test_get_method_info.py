import pytest
from typing import Any

from jet.utils.inspect_utils import get_method_info


class TestGetMethodInfoExtended:
    def test_get_method_info_with_regular_function(self):
        # Given: A regular function with type hints
        def add_numbers(a: int, b: float) -> float:
            """Add two numbers and return the result."""
            return a + b

        expected = {
            "name": "add_numbers",
            "parameters": "a: int, b: float",
            "return_type": "float",
            "docstring": "Add two numbers and return the result.",
        }

        # When: We call get_method_info with a regular function
        result = get_method_info(add_numbers)

        # Then: The result matches the expected dictionary
        assert result == expected

    def test_get_method_info_with_static_method(self):
        # Given: A static method in a class
        class MathService:
            @staticmethod
            def multiply(x: int, y: int) -> int:
                """Multiply two integers."""
                return x * y

        expected = {
            "name": "multiply",
            "parameters": "x: int, y: int",
            "return_type": "int",
            "docstring": "Multiply two integers.",
        }

        # When: We call get_method_info with a static method
        result = get_method_info(MathService.multiply)

        # Then: The result matches the expected dictionary
        assert result == expected

    def test_get_method_info_with_lambda(self):
        # Given: A lambda function (no type hints or docstring)
        def lambda_func(x, y): return x + y
        expected = {
            "name": "<lambda>",  # Lambdas have a generic name
            "parameters": "x: Any, y: Any",  # No type hints, so Any is used
            "return_type": "Any",  # No return type hint
            "docstring": "No docstring available",
        }

        # When: We call get_method_info with a lambda
        result = get_method_info(lambda_func)

        # Then: The result matches the expected dictionary
        assert result == expected
