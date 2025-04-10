import textwrap
import os
import string
from typing import Any, Dict
from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import chat
from jet.transformers.json_parsers import parse_json
from jet.utils.markdown import extract_block_content
from jet.validation.python_validation import validate_python_syntax

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "template_strings")


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


def _run_chat(prompt: str, model: OLLAMA_MODEL_NAMES) -> Any:
    response = chat(model, prompt, temperature=0.0)
    content = response['message']['content']
    result = extract_block_content(content)
    return result


def generate_json_schema(context: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b") -> Dict:
    prompt = _generate_prompt("Generate_JSON_Schema.md", context=context)
    json_result = _run_chat(prompt, model)
    result = parse_json(json_result)
    return result


def generate_pydantic_models(context: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b") -> str:
    prompt = _generate_prompt("Generate_Pydantic_Models.md", context=context)
    python_code = _run_chat(prompt, model)
    python_code = textwrap.dedent(python_code).strip()
    errors = validate_python_syntax(python_code)
    if errors:
        raise ValueError(f"Syntax errors detected:\n" + "\n".join(
            [f"File: {error['file']}, Line: {error['line']}, Error: {error['message']}, Code: {error['code']}" for error in errors]
        ))
    return python_code
