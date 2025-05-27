import os
import json
from typing import List, Optional, TypedDict
from jet.llm.mlx.base import MLX
from jet.llm.mlx.client import MLXLMClient
from jet.llm.mlx.mlx_types import ModelKey
from jet.logger import logger
from jet.transformers.formatters import format_json
MODEL_PATH: ModelKey = "llama-3.2-3b-instruct-4bit"
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Generate entity labels for NER from these texts: ['that time i got reincarnated as a slime', 'rimuru tempest builds a monster nation in a new world', 'isekai, fantasy, nation building, slime', 'isekai anime with monster protagonists']"
    },
    {
        "role": "assistant",
        "content": "{\"labels\": [\"Anime Title\", \"Character\", \"World\", \"Genre\", \"Activity\"]}"
    },
    {
        "role": "user",
        "content": "Generate entity labels for NER from these texts: ['OpenAI - Artificial Intelligence Research', 'OpenAI develops advanced AI models like ChatGPT.', 'artificial intelligence, machine learning, chatbot, research', 'best AI tools for developers']"
    },
    {
        "role": "assistant",
        "content": "{\"labels\": [\"Company\", \"Technology\", \"Tool\", \"Genre\", \"Profession\"]}"
    }
]
SYSTEM_PROMPT = "You are an AI that generates concise, singular noun labels for Named Entity Recognition (NER) compatible with GLiNER. Labels should represent entity types (e.g., 'Anime Title', 'Character', 'Genre') derived from the provided texts. Return a JSON list of unique labels under the key 'labels'."
MAX_TOKENS = 200


class LabelExtractionResult(TypedDict):
    labels: List[str]


def build_prompt(texts: List[str]) -> dict:
    """Wrap list of texts in the expected format."""
    texts_str = json.dumps(texts, ensure_ascii=False)
    return {"role": "user", "content": f"Generate entity labels for NER from these texts: {texts_str}"}


def parse_response(response: str) -> Optional[LabelExtractionResult]:
    """Attempt to parse a valid JSON response and validate labels."""
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start == -1 or json_end == -1:
            logger.error("No JSON content found.")
            return None
        parsed = json.loads(response[json_start:json_end])
        if "labels" not in parsed or not isinstance(parsed["labels"], list):
            logger.error("Invalid or missing 'labels' key.")
            return None
        # Validate labels: ensure they are singular nouns or noun phrases
        valid_labels = [
            label for label in parsed["labels"]
            if isinstance(label, str) and len(label.split()) <= 2 and label[0].isupper()
        ]
        return {"labels": valid_labels}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding error: {e}")
        return None


def generate_labels(
    texts: List[str],
    model_path: ModelKey = MODEL_PATH,
    system_prompt: str = SYSTEM_PROMPT,
    few_shot_examples: List[dict] = FEW_SHOT_EXAMPLES,
    max_tokens: int = MAX_TOKENS,
    max_labels: Optional[int] = None
) -> List[str]:
    """Generates unique entity labels for NER from a list of texts."""
    client = MLX(model_path, seed=42)
    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        build_prompt(texts)
    ]
    logger.debug("Streaming Label Generation Response:")
    full_response = ""
    for response in client.stream_chat(
        messages=messages,
        model=model_path,
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=1.0,
        repetition_penalty=1.2,
        stop=["\n\n"],
    ):
        if response["choices"]:
            content = response["choices"][0].get(
                "message", {}).get("content", "")
            full_response += content
            logger.success(content, flush=True)
            if response["choices"][0]["finish_reason"]:
                logger.newline()
    parsed_response = parse_response(full_response)
    if parsed_response is None:
        logger.error("Failed to generate valid labels.")
        return []
    labels = parsed_response["labels"]
    if max_labels is not None:
        labels = labels[:max_labels]
    return labels
