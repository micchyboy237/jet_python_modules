import json
from typing import List, Optional, TypedDict
from jet.transformers.object import make_serializable
from jsonschema import Draft7Validator
from jet.validation import ValidationResponse


def format_error(error) -> str:
    """Format the error message to include the path to the error."""
    error_path = ".".join([str(p) for p in error.path])
    return f"{error_path}: {error.message}"


def schema_validate_json(json_string: str | dict, schema: Optional[dict] = None) -> ValidationResponse:
    if not isinstance(json_string, str):
        json_string = json.dumps(json_string)

    if isinstance(schema, str):
        schema = make_serializable(schema)

    try:
        data = json.loads(json_string)

        # If schema is provided, perform validation
        if schema:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(data))

            if errors:
                error_messages = [format_error(e) for e in errors]
                return ValidationResponse(is_valid=False, data=None, errors=error_messages)

        # If no schema is provided, simply return the parsed data
        return ValidationResponse(is_valid=True, data=data, errors=None)

    except json.JSONDecodeError as e:
        return ValidationResponse(is_valid=False, data=None, errors=[str(e)])

    except Exception as e:
        return ValidationResponse(is_valid=False, data=None, errors=[str(e)])
