import json
import os
import string
import textwrap
from typing import Any, Dict, Type

from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_TYPES
from jet.data.base import BaseModel, convert_json_schema_to_model_type
from jet.file.utils import load_file
from jet.transformers.json_parsers import parse_json
from jet.utils.markdown import extract_block_content
from jet.validation.json_schema_validator import schema_validate_json
from jet.validation.python_validation import validate_python_syntax

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "template_strings")
_DEFAULT_MODEL: LLAMACPP_LLM_TYPES = "qwen3-instruct-2507:4b"
_DEFAULT_SEED = 42


def extract_template_variables(template: str) -> set:
    formatter = string.Formatter()
    return {
        field_name for _, field_name, _, _ in formatter.parse(template) if field_name
    }


def _load_template(name: str) -> str:
    return load_file(os.path.join(_TEMPLATE_DIR, name))


def _generate_prompt(template_file: str, **kwargs: Any) -> str:
    template = _load_template(template_file)
    required_vars = extract_template_variables(template)
    provided_vars = set(kwargs.keys())

    missing = required_vars - provided_vars
    if missing:
        raise ValueError(f"Missing template variables: {missing}")

    return template.format(**kwargs)


def _get_llm(model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL) -> LlamacppLLM:
    return LlamacppLLM(
        model=model,
        # verbose=False,
        # max_retries=2,
    )


def _run_chat(
    prompt_or_messages: str | list[dict],
    model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL,
    system: str | None = None,
    context: str | None = None,  # New optional parameter
    **kwargs: Any,
) -> str:
    llm = _get_llm(model)

    if isinstance(prompt_or_messages, str):
        user_content = prompt_or_messages

        # Only apply context formatting when we have a simple string prompt
        if context:
            user_content = f"""Context / Additional Information:
{context}

---

User Query / Task:
{prompt_or_messages}"""

        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [{"role": "user", "content": user_content}]
    else:
        # Full message list provided → do NOT modify it
        messages = prompt_or_messages

    response = llm.chat(messages, temperature=0.0, **kwargs)

    if isinstance(response, str):
        content = response
    else:
        # If streaming, concatenate all responses
        content = "".join(response)

    return extract_block_content(content)


def generate_json_schema(
    query: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> Dict:
    prompt = _generate_prompt("Generate_JSON_Schema.md", query=query)
    json_result = _run_chat(prompt, model=model)
    result = parse_json(json_result)
    return result


def generate_browser_query_json_schema(
    query: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> Dict:
    system = _generate_prompt("System_Browser_Query_JSON_Schema.md")
    json_result = _run_chat(query, model=model, system=system)
    result = parse_json(json_result)
    return result


def generate_browser_query_context_json_schema(
    query: str, context: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> Dict:
    system = _generate_prompt("System_Browser_Query_Context_JSON_Schema.md")
    json_result = _run_chat(query, model=model, context=context, system=system)
    result = parse_json(json_result)
    return result


def generate_json_schema_sample(
    json_schema: str | dict, query: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> Dict:
    if not isinstance(json_schema, str):
        json_schema = json.dumps(json_schema, indent=2)
    prompt = _generate_prompt(
        "Generate_JSON_Schema_Sample.md", json_schema=json_schema, query=query
    )
    json_result = _run_chat(prompt, model=model)
    result = parse_json(json_result)

    validation_result = schema_validate_json(result, json_schema)
    # Validate generated sample with schema
    if not validation_result["is_valid"]:
        errors = validation_result["errors"]
        raise ValueError(
            "JSON schema and data syntax errors detected:\n"
            + json.dumps(errors, indent=2)
        )

    return validation_result["data"]


def generate_pydantic_models(
    context: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> str:
    prompt = _generate_prompt("Generate_Pydantic_Models.md", context=context)
    python_code = _run_chat(prompt, model=model)
    python_code = textwrap.dedent(python_code).strip()

    # Validate the Python syntax before executing it
    errors = validate_python_syntax(python_code)
    if errors:
        raise ValueError(
            "Syntax errors detected:\n"
            + "\n".join(
                [
                    f"File: {error['file']}, Line: {error['line']}, Error: {error['message']}, Code: {error['code']}"
                    for error in errors
                ]
            )
        )

    return python_code


def generate_field_descriptions(
    query: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> str:
    """
    Generate a short, clear description of the expected data fields
    for the given query. Used by Pydantic models and JSON schemas.
    """
    prompt = _generate_prompt("Generate_Field_Descriptions.md", query=query)
    field_desc = _run_chat(prompt, model=model)
    return field_desc.strip()


def generate_output_class(
    query: str, model: LLAMACPP_LLM_TYPES = _DEFAULT_MODEL
) -> Type[BaseModel]:
    """
    Generate output model structure
    """
    json_schema = generate_browser_query_json_schema(query, model=model)
    DynamicModel = convert_json_schema_to_model_type(json_schema)
    output_cls = DynamicModel
    return output_cls
