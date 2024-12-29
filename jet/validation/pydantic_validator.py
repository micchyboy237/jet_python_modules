import json
from typing import Optional, List, TypedDict
from pydantic import BaseModel, ValidationError
from jet.validation import ValidationResponse


def format_error(error: ValidationError) -> str:
    """Format the error message to include the path to the error."""
    error_path = ".".join([str(p) for p in error["loc"]])
    error_chunks = [error["msg"]]
    if error_path:
        error_chunks.insert(0, error_path)
    return ": ".join(error_chunks)


def pydantic_validate_json(json_string: str, model: BaseModel) -> ValidationResponse:
    """Validate a Pydantic BaseModel instance and return a formatted response."""
    try:
        model.model_validate_json(json_string, strict=True)

        data = json.loads(json_string)

        return ValidationResponse(is_valid=True, data=data, errors=None)

    except ValidationError as e:
        errors = e.errors().copy()
        for error in errors:
            error.pop('url', None)
        error_messages = [format_error(e) for e in errors]
        return ValidationResponse(is_valid=False, data=None, errors=error_messages)

    except Exception as e:
        return ValidationResponse(is_valid=False, data=None, errors=[str(e)])
