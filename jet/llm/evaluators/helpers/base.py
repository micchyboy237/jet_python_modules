import json
import re
from typing import Optional
from llama_index.core.evaluation.base import EvaluationResult as BaseEvaluationResult
from llama_index.core.bridge.pydantic import Field
from jet.utils.markdown import extract_json_block_content


class EvaluationResult(BaseEvaluationResult):
    excerpts: list[str] = Field(
        default=[],
        description="A list of relevant excerpts from the context that directly answer the query. These are specific sections or pieces of the context that provide concrete, actionable answers to the user's query. If no relevant excerpts exist, an empty list should be provided."
    )


def default_parser_function(output_str: str) -> tuple[Optional[float], Optional[str]]:
    # Pattern to match the feedback and response
    # This pattern looks for any text ending with '[RESULT]' followed by a number
    pattern = r"([\s\S]+)(?:\[RESULT\]\s*)([\d.]+)"

    # Using regex to find all matches
    result = re.search(pattern, output_str)

    # Check if any match is found
    if result:
        # Assuming there's only one match in the text, extract feedback and response
        feedback, score = result.groups()
        score = float(score) if score is not None else score
        return score, feedback.strip()
    else:
        return None, None


def parse_excerpts(output_str: str) -> list[str]:
    """
    Extracts the excerpts from the output string, which should be in JSON array format.

    Args:
        output_str (str): The output string from the LLM containing the excerpts.

    Returns:
        Optional[list[str]]: A list of excerpts if found, or None if no excerpts exist.
    """
    # Pattern to match the [EXCERPTS] block surrounded by ```json
    pattern = r"\[EXCERPTS\] ```json\n(\[.*?\])\n```"

    # Search for the pattern in the output string
    result = re.search(pattern, output_str, re.DOTALL)

    # If the pattern is found, try to parse the JSON content
    if result:
        excerpts_json = result.group(1)  # Extract the JSON part
        try:
            # Parse the JSON string into a Python list
            excerpts = json.loads(excerpts_json)
            return excerpts
        except json.JSONDecodeError:
            # Return an empty list if JSON parsing fails
            return []
    else:
        # Return an empty list if no excerpts are found
        return []
