import json
from typing import Optional, List, TypedDict
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.class_utils import get_class_name
from jet.utils.object import get_values_by_paths
from pydantic import BaseModel, ValidationError
from jet.validation import ValidationResponse


def get_error_path(loc: list[str]) -> str:
    return ".".join([str(p) for p in loc])


def format_errors(errors: list[dict], data: dict) -> list[str]:
    """Format the error message to include the path to the error."""
    error_paths = [get_error_path(error.get("loc", []))
                   for error in errors]
    error_path_values = get_values_by_paths(data, error_paths)

    error_messages = []
    for idx, error in enumerate(errors):
        error_path = get_error_path(error.get("loc", []))
        data_value = error_path_values[idx]

        msg = f"\"{error_path}\" <{get_class_name(data_value)}> should be a valid {error['type'].replace("_", " ")}"
        error_messages.append(msg)
    return error_messages


def pydantic_validate_json(json_string: str, model: BaseModel) -> ValidationResponse:
    """Validate a Pydantic BaseModel instance and return a formatted response."""
    try:
        data = json.loads(json_string)

        model.model_validate_json(json_string, strict=True)

        return ValidationResponse(is_valid=True, data=data, errors=None)

    except ValidationError as e:
        errors = e.errors().copy()
        error_messages = format_errors(errors, data)
        return ValidationResponse(is_valid=False, data=None, errors=error_messages)

    except Exception as e:
        return ValidationResponse(is_valid=False, data=None, errors=[str(e)])


# Example Usage
if __name__ == "__main__":
    class Address(BaseModel):
        country: str

    class User(BaseModel):
        name: str
        age: Optional[int]
        address: Address

    # With invalid values
    json_data = {
        "name": None,
        "age": "Thirty",
        "address": {
            "country": 0
        }
    }
    json_str = json.dumps(json_data)

    response = pydantic_validate_json(json_str, User)
    expected = [
        "\"name\" <NoneType> should be a valid string type",
        "\"age\" <str> should be a valid int type",
        "\"address.country\" <int> should be a valid string type"
    ]

    assert response["is_valid"] is False
    assert response["errors"] == expected
    logger.newline()
    # Should capture all errors
    logger.success(format_json(response["errors"]))
