import os
import json
from typing import List, Optional, TypedDict

from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import MLXLMClient
from jet.llm.mlx.mlx_types import ModelKey
from jet.logger import logger
from jet.transformers.formatters import format_json


MODEL_PATH: ModelKey = "llama-3.2-1b-instruct-4bit"
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "What are the key concepts in this sentence: 'Neural networks are powerful for image recognition'?"
    },
    {
        "role": "assistant",
        "content": "{\"tags\": [\"neural networks\", \"image recognition\"]}"
    },
    {
        "role": "user",
        "content": "Extract important keywords: 'Python is a great language for data science.'"
    },
    {
        "role": "assistant",
        "content": "{\"tags\": [\"Python\", \"data science\"]}"
    }
]
SYSTEM_PROMPT = """
You are an AI assistant that follows instructions. You read and understand both structured and unstructured web scraped data. You analyze content and extract meaningful concepts. You generate descriptive tags based on the content. You output the tags in JSON format using an object with a "tags" array of strings. You output only the JSON block without any additional text.
"""
MAX_TOKENS = 150

MODEL: ModelKey = "llama-3.2-1b-instruct-4bit"

mlx = MLX(MODEL)


class TagExtractionResult(TypedDict):
    tags: List[str]


def build_prompt(texts: List[str]) -> dict:
    """Wrap list of texts in the expected format."""
    texts_str = json.dumps(texts, ensure_ascii=False)
    return {"role": "user", "content": texts_str}


def parse_response(response: str) -> TagExtractionResult:
    """Attempt to parse a valid JSON response."""
    empty_result: TagExtractionResult = {"tags": []}
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            logger.error("No JSON content found.")
            return empty_result
        parsed = json.loads(response[json_start:json_end])
        if "tags" not in parsed or not isinstance(parsed["tags"], list):
            logger.error("Invalid or missing 'tags' key.")
            return empty_result
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return empty_result


def generate_tags(
    texts: List[str],
    model_path: ModelKey = MODEL_PATH,
    system_prompt: str = SYSTEM_PROMPT,
    few_shot_examples: List[dict] = FEW_SHOT_EXAMPLES,
    max_tokens: int = MAX_TOKENS
) -> List[str]:
    """Performs streaming tag generation with specified configuration."""
    logger.debug("Streaming Tag Generation Response:")
    response = ""

    context = mlx.filter_docs(texts)

    for chunk in mlx.stream_chat(
        context,
        model=model_path,
        max_tokens=max_tokens,
        temperature=0.7,
        system_prompt=system_prompt,
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)

        if chunk["choices"][0]["finish_reason"]:
            logger.newline()
            # logger.orange(format_json(response))

    result = parse_response(response)
    return result["tags"]
