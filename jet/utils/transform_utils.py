import json
from pydantic import BaseModel, Field
from typing import Any, Dict

# Function to convert JSON schema to Pydantic model


def json_schema_to_pydantic(json_schema: str) -> BaseModel:
    # Parse the JSON schema string into a dictionary
    schema = json.loads(json_schema)

    # Create a dynamic dictionary to store the model fields
    model_fields = {}

    # Map JSON schema types to Pydantic types
    type_mapping = {
        'string': str,
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'number': float,
        'null': Any,  # For nullable fields
    }

    for property_name, property_definition in schema.get('properties', {}).items():
        # Default to str if type is not recognized
        field_type = type_mapping.get(property_definition.get('type'), str)
        description = property_definition.get('description', '')

        # Use Pydantic's Field to add description and other attributes
        model_fields[property_name] = (
            field_type, Field(..., description=description))

    # Create a dynamic Pydantic model
    return type('DynamicModel', (BaseModel,), model_fields)


if __name__ == "__main__":
    # Example JSON Schema string
    json_schema_string = '''
    {
    "type": "object",
    "properties": {
        "title": {
        "type": "string",
        "description": "The exact title of the anime."
        },
        "document_number": {
        "type": "integer",
        "description": "The number of the document that includes this anime."
        },
        "release_year": {
        "type": "integer",
        "description": "The most recent known release year of the anime."
        }
    },
    "required": ["title", "document_number"]
    }
    '''

    # Convert the JSON Schema to a Pydantic model
    DynamicModel = json_schema_to_pydantic(json_schema_string)

    # Example usage of the generated model
    model_instance = DynamicModel(
        title="Naruto", document_number=1, release_year=2002)

    print(model_instance)
