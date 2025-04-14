import json
from pydantic import create_model, BaseModel as PydanticBaseModel
from typing import Type, TypedDict, Optional, Dict, Any, List, Union


class BaseModel(PydanticBaseModel):
    # 👈 class-level attribute to store schema
    _original_json_schema: Dict[str, Any] = None

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> Dict[str, Any]:
        # if hasattr(cls, "_original_json_schema") and cls._original_json_schema:
        #     return remove_dollar_keys(cls._original_json_schema)
        return super().model_json_schema(*args, **kwargs)


def remove_dollar_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: remove_dollar_keys(v)
            for k, v in obj.items()
            if not k.startswith("$")
        }
    elif isinstance(obj, list):
        return [remove_dollar_keys(i) for i in obj]
    return obj


def map_json_type_to_python(
    json_type: Union[str, list[str]],
    field_schema: Dict[str, Any],
    field_name: str = "Nested",
    nested: bool = False
) -> Any:
    if isinstance(json_type, list):
        non_null_types = [t for t in json_type if t != "null"]
        if len(non_null_types) == 1:
            base_type = map_json_type_to_python(
                non_null_types[0], field_schema, field_name, nested)
            return Optional[base_type]
        return Any

    if json_type == "string":
        return str
    elif json_type == "number":
        return float
    elif json_type == "integer":
        return int
    elif json_type == "boolean":
        return bool
    elif json_type == "null":
        return type(None)
    elif json_type == "object":
        if nested:
            sub_model = create_dynamic_model(
                field_schema, model_name=field_name.capitalize() + "Model", nested=nested)
            return sub_model
        else:
            return Dict[str, Any]
    elif json_type == "array":
        if nested:
            items_schema = field_schema.get("items", {})
            items_type = map_json_type_to_python(
                items_schema.get("type", "any"),
                items_schema,
                field_name + "Item",
                nested
            )
        else:
            items_type = map_json_type_to_python(
                field_schema.get("items", {}).get("type", "any"),
                field_schema.get("items", {})
            )

        return list[items_type]
    return Any


def create_dynamic_model(
    json_schema: str | Dict[str, Any],
    *,
    model_name: str = "DynamicModel",
    nested: bool = False,
    base_model: Type[BaseModel] = BaseModel
) -> Type[BaseModel]:
    if isinstance(json_schema, str):
        json_schema = json.loads(json_schema)

    model_fields = {}
    required_fields = set(json_schema.get("required", []))
    properties = json_schema.get("properties", {})

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "any")
        python_type = map_json_type_to_python(
            json_type, field_schema, field_name, nested)

        if field_name in required_fields:
            model_fields[field_name] = (python_type, ...)
        else:
            model_fields[field_name] = (Optional[python_type], None)

    model = create_model(model_name, **model_fields, __base__=base_model)
    model._original_json_schema = json_schema  # 👈 Attach original schema
    return model


def convert_json_schema_to_model_type(json_schema: str | Dict[str, Any]) -> Type[BaseModel]:
    DynamicModel = create_dynamic_model(json_schema)
    return DynamicModel


def convert_json_schema_to_model_instance(
    data: Dict[str, Any], json_schema: str | Dict[str, Any]
) -> BaseModel:
    DynamicModel = create_dynamic_model(json_schema)
    return DynamicModel(**data)


class SchemaInfo(TypedDict):
    path: str
    title: Optional[str]
    description: Optional[str]


def extract_titles_descriptions(schema: Dict[str, Any], path: str = "", result: Optional[List[SchemaInfo]] = None) -> List[SchemaInfo]:
    if result is None:
        result = []

    # Handle schema types
    if isinstance(schema, dict):
        # Extract title and description at current level
        title = schema.get("title")
        description = schema.get("description")
        if title or description:
            result.append({"path": path, "title": title,
                          "description": description})

        # Handle properties for object schemas
        if "properties" in schema:
            for prop, subschema in schema.get("properties", {}).items():
                new_path = f"{path}.{prop}" if path else prop
                extract_titles_descriptions(subschema, new_path, result)

        # Handle items for array schemas
        if "items" in schema:
            new_path = f"{path}.items" if path else "items"
            extract_titles_descriptions(schema["items"], new_path, result)

        # Handle additionalProperties if it's a schema
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            new_path = f"{path}.additionalProperties" if path else "additionalProperties"
            extract_titles_descriptions(
                schema["additionalProperties"], new_path, result)

        # Handle allOf, anyOf, oneOf
        for combiner in ["allOf", "anyOf", "oneOf"]:
            if combiner in schema:
                for i, subschema in enumerate(schema[combiner]):
                    new_path = f"{path}.{combiner}[{i}]" if path else f"{combiner}[{i}]"
                    extract_titles_descriptions(subschema, new_path, result)

        # Handle $ref (optional, depends on whether you want to resolve references)
        if "$ref" in schema:
            # If you have a resolver, you can resolve $ref here
            pass

    return result
