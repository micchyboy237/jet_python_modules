from typing import Optional, TypedDict


class ValidationResponse(TypedDict):
    is_valid: bool
    data: Optional[dict]
    errors: Optional[list[str]]
