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
            "body": "return a + b"
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
            "body": "return x * y"
        }

        # When: We call get_method_info with a static method
        result = get_method_info(MathService.multiply)

        # Then: The result matches the expected dictionary
        assert result == expected

    def test_get_method_info_with_lambda(self):
        # Given: A lambda function (no type hints or docstring)
        def lambda_func(x, y): return x + y  # Use actual lambda expression
        expected = {
            "name": "lambda_func",
            "parameters": "x: Any, y: Any",
            "return_type": "Any",
            "docstring": "No docstring available",
            "body": "Source code unavailable"
        }

        # When: We call get_method_info with a lambda
        result = get_method_info(lambda_func)

        # Then: The result matches the expected dictionary
        assert result == expected

    def test_get_method_info_with_body(self):
        # Given: A sample method with parameters, return type, docstring, and body
        def sample_method(x: int, y: str) -> bool:
            """Sample method that returns a boolean."""
            return x > 0 and len(y) > 0

        # When: We call get_method_info on the sample method
        result = get_method_info(sample_method)

        # Then: The result should contain the correct method info including the body
        expected = {
            "name": "sample_method",
            "parameters": "x: int, y: str",
            "return_type": "bool",
            "docstring": "Sample method that returns a boolean.",
            "body": "return x > 0 and len(y) > 0"
        }
        assert result == expected
