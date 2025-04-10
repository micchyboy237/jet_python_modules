import os
from typing import Dict
from jet.file.utils import load_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.llm.ollama.base import ChatResponse, chat
from jet.transformers.json_parsers import parse_json
from jet.utils.markdown import extract_json_block_content


def get_chat_func(prompt: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b"):
    def run_chat():
        return chat(model, prompt, temperature=0.0)
    return run_chat


def generate_json_schema_prompt(context: str) -> str:
    template_str: str = load_file(
        f"{os.path.dirname(__file__)}/template_strings/Generate_JSON_Schema.md")
    prompt = template_str.format(context=context)
    return prompt


def generate_json_schema(context: str, model: OLLAMA_MODEL_NAMES = "gemma3:1b") -> Dict:
    prompt = generate_json_schema_prompt(context)
    response = chat(prompt, model)
    content = response['message']['content']
    json_result = extract_json_block_content(content)
    result = parse_json(json_result)
    return result
