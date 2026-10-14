"""
Core math utilities for the test repository.
This file serves as ground truth for the evaluation harness.
"""


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two integers."""
    return a + b


def calculate_product(a: int, b: int) -> int:
    """Calculate the product of two integers."""
    return a * b


def validate_positive(value: int) -> bool:
    """Check if value is strictly positive."""
    return value > 0
