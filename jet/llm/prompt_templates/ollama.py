import json
import textwrap
import os
import string
from typing import Any, Dict, Type
from jet.data.base import BaseModel, convert_json_schema_to_model_type
from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import chat
from jet.transformers.json_parsers import parse_json
from jet.utils.markdown import extract_block_content
from jet.validation.json_schema_validator import schema_validate_json
from jet.validation.main.json_validation import validate_json
from jet.validation.python_validation import validate_python_syntax
from jet.validation.validation_types import ValidationResponse

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "template_strings")
_DEFAULT_MODEL = "gemma3:4b"
_DEFAULT_SEED = 42


def extract_template_variables(template: str) -> set:
    formatter = string.Formatter()
    return {field_name for _, field_name, _, _ in formatter.parse(template) if field_name}


def _load_template(name: str) -> str:
    return load_file(os.path.join(_TEMPLATE_DIR, name))


def _generate_prompt(template_file: str, **kwargs) -> str:
    template = _load_template(template_file)
    required_vars = extract_template_variables(template)
    provided_vars = set(kwargs.keys())

    missing = required_vars - provided_vars
    if missing:
        raise ValueError(f"Missing template variables: {missing}")

    return template.format(**kwargs)


def _run_chat(prompt: str, model: OLLAMA_MODEL_NAMES, **kwargs) -> Any:
    response = chat(prompt, model, temperature=0.0,
                    seed=_DEFAULT_SEED, **kwargs)
    content = response['message']['content']
    result = extract_block_content(content)
    return result


def generate_json_schema(query: str, model: OLLAMA_MODEL_NAMES = _DEFAULT_MODEL) -> Dict:
    prompt = _generate_prompt("Generate_JSON_Schema.md", query=query)
    json_result = _run_chat(prompt, model)
    result = parse_json(json_result)
    return result


def generate_browser_query_json_schema(query: str, model: OLLAMA_MODEL_NAMES = _DEFAULT_MODEL) -> Dict:
    # prompt = _generate_prompt(
    #     "Generate_Browser_Query_JSON_Schema.md", browser_query=query)
    system = _generate_prompt("System_Browser_Query_JSON_Schema.md")
    json_result = _run_chat(query, model, system=system)
    result = parse_json(json_result)
    return result


def generate_browser_query_context_json_schema(query: str, context: str, model: OLLAMA_MODEL_NAMES = _DEFAULT_MODEL) -> Dict:
    # prompt = _generate_prompt(
    #     "Generate_Browser_Query_JSON_Schema.md", browser_query=query)
    system = _generate_prompt("System_Browser_Query_Context_JSON_Schema.md")
    json_result = _run_chat(query, model, context=context, system=system)
    result = parse_json(json_result)
    return result


def generate_json_schema_sample(json_schema: str | dict, query: str, model: OLLAMA_MODEL_NAMES = _DEFAULT_MODEL) -> Dict:
    if not isinstance(json_schema, str):
        json_schema = json.dumps(json_schema, indent=2)
    prompt = _generate_prompt(
        "Generate_JSON_Schema_Sample.md", json_schema=json_schema, query=query)
    json_result = _run_chat(prompt, model)
    result = parse_json(json_result)

    validation_result = validate_json(result, json_schema)
    # Validate generated sample with schema
    if not validation_result["is_valid"]:
        errors = validation_result["errors"]
        raise ValueError(f"JSON schema and data syntax errors detected:\n" +
                         json.dumps(errors, indent=2))

    return validation_result["data"]


def generate_pydantic_models(context: str, model: OLLAMA_MODEL_NAMES = _DEFAULT_MODEL) -> str:
    prompt = _generate_prompt("Generate_Pydantic_Models.md", context=context)
    python_code = _run_chat(prompt, model)
    python_code = textwrap.dedent(python_code).strip()

    # Validate the Python syntax before executing it
    errors = validate_python_syntax(python_code)
    if errors:
        raise ValueError(f"Syntax errors detected:\n" + "\n".join(
            [f"File: {error['file']}, Line: {error['line']}, Error: {error['message']}, Code: {error['code']}" for error in errors]
        ))

    return python_code


def generate_output_class(query: str, model: OLLAMA_MODEL_NAMES) -> Type[BaseModel]:
    """
    Generate output model structure
    """
    json_schema = generate_browser_query_json_schema(query, model)
    DynamicModel = convert_json_schema_to_model_type(json_schema)
    output_cls = DynamicModel
    return output_cls
