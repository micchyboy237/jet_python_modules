import re
from typing import Dict, List, TypedDict, Optional
import json
from typing import Optional
from llama_index.core.evaluation.base import EvaluationResult as BaseEvaluationResult
from llama_index.core.bridge.pydantic import BaseModel, Field


class Comment(BaseModel):
    text: str
    score: float


class EvaluationDetails(BaseModel):
    comments: List[Comment] = []


class EvaluationResult(BaseEvaluationResult):
    details: EvaluationDetails = Field(
        default_factory=lambda: EvaluationDetails(comments=[]),
        description="Detailed evaluation of the response, including specific comments."
    )
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


def parse_feedback(feedback_str: Optional[str]) -> EvaluationDetails:
    """
    Parses a feedback string to extract the score and individual comments.

    Args:
        feedback_str: The string containing the feedback, score, and comments.
                    This is now optional.

    Returns:
        An instance of EvaluationDetails.
    """
    results = EvaluationDetails()
    if not feedback_str:  # Handle the case where feedback_str is None or empty
        return results

    feedback_match = re.search(
        r"Feedback:\n(.*?)\n\n\[RESULT\]", feedback_str, re.DOTALL)

    if feedback_match:
        comments_str = feedback_match.group(1).strip()
        comment_lines = [line.strip()
                         for line in comments_str.split('\n') if line.strip()]
        for line in comment_lines:
            parts = line.split('(Score: ')
            text = parts[0].strip()
            score = None
            if len(parts) > 1:
                score_str = parts[1].rstrip(')')
                try:
                    score = float(score_str)
                except ValueError:
                    score = None  # Handle cases where the score isn't a valid number
            results.comments.append(Comment(text=text, score=score))

    return results  # Return EvaluationDetails


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
