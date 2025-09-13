from typing import TypedDict


class ValidationResult(TypedDict):
    passed: bool
    error: str | None
