import os
import string
from typing import Dict
from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import chat
from jet.transformers.json_parsers import parse_json
from jet.utils.markdown import extract_json_block_content

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


def _run_chat(prompt: str, model: OLLAMA_MODEL_NAMES) -> Dict:
    response = chat(model, prompt, temperature=0.0)
    content = response['message']['content']
    json_result = extract_json_block_content(content)
    return parse_json(json_result)


def generate_json_schema(context: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b") -> Dict:
    prompt = _generate_prompt("Generate_JSON_Schema.md", context=context)
    return _run_chat(prompt, model)


def generate_pydantic_models(context: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b") -> Dict:
    prompt = _generate_prompt("Generate_Pydantic_Models.md", context=context)
    return _run_chat(prompt, model)
