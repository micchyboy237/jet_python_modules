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
            "description": "Add two numbers and return the result.",
            "parameters": {
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer", "description": "The a parameter"},
                    "b": {"type": "number", "description": "The b parameter"}
                }
            }
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
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "integer", "description": "The x parameter"},
                    "y": {"type": "integer", "description": "The y parameter"}
                }
            }
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
            "description": "No description available",
            "parameters": {
                "type": "object",
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "any", "description": "The x parameter"},
                    "y": {"type": "any", "description": "The y parameter"}
                }
            }
        }

        # When: We call get_method_info with a lambda
        result = get_method_info(lambda_func)

        # Then: The result matches the expected dictionary
        assert result == expected

    def test_get_method_info_with_body(self):
        # Given: A sample method with parameters, return type, and docstring
        def sample_method(x: int, y: str) -> bool:
            """Sample method that returns a boolean."""
            return x > 0 and len(y) > 0

        # When: We call get_method_info on the sample method
        result = get_method_info(sample_method)

        # Then: The result should contain the correct method info
        expected = {
            "name": "sample_method",
            "description": "Sample method that returns a boolean.",
            "parameters": {
                "type": "object",
                "required": ["x", "y"],
                "properties": {
                    "x": {"type": "integer", "description": "The x parameter"},
                    "y": {"type": "string", "description": "The y parameter"}
                }
            }
        }
        assert result == expected
