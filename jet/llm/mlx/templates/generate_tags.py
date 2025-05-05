import os
import json
from typing import List, Optional, TypedDict

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
SYSTEM_PROMPT = "You are an AI that extracts key tags or keywords from user text and returns them as a JSON list under the key 'tags'."
MAX_TOKENS = 150


class TagExtractionResult(TypedDict):
    tags: List[str]


def build_prompt(user_prompt: str) -> dict:
    """Wrap user input in the expected format."""
    return {"role": "user", "content": f"Input:\n```\n{user_prompt}\n```"}


def parse_response(response: str) -> Optional[TagExtractionResult]:
    """Attempt to parse a valid JSON response."""
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            logger.error("No JSON content found.")
            return None
        parsed = json.loads(response[json_start:json_end])
        if "tags" not in parsed or not isinstance(parsed["tags"], list):
            logger.error("Invalid or missing 'tags' key.")
            return None
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return None


def generate_tags(
    user_input: str,
    model_path: ModelKey = MODEL_PATH,
    system_prompt: str = SYSTEM_PROMPT,
    few_shot_examples: List[dict] = FEW_SHOT_EXAMPLES,
    max_tokens: int = MAX_TOKENS
) -> Optional[TagExtractionResult]:
    """Performs streaming tag generation with specified configuration."""
    client = MLXLMClient(model_path)

    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        build_prompt(user_input)
    ]

    logger.debug("Streaming Tag Generation Response:")
    full_response = ""

    for response in client.stream_chat(
        messages=messages,
        model=model_path,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=["\n\n"]
    ):
        if response["choices"]:
            content = response["choices"][0].get(
                "message", {}).get("content", "")
            full_response += content
            logger.success(content, flush=True)

            if response["choices"][0]["finish_reason"]:
                logger.newline()
                # logger.orange(format_json(response))

    return parse_response(full_response)
