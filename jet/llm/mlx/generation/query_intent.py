from typing import Literal, TypedDict
from jet.llm.mlx.base import MLX
from jet.llm.mlx.helpers.base import get_system_date_prompt
from jet.models.model_types import LLMModelType


class QueryIntent(TypedDict):
    intent: Literal["informational", "navigational",
                    "transactional", "commercial_investigation"]
    confidence: float


def get_query_intent(original_query: str, model: LLMModelType = "qwen3-1.7b-4bit") -> QueryIntent:
    """
    Determines the intent of a user's query, classifying it as informational, navigational, transactional,
    or commercial investigation, with a confidence score.

    Args:
        original_query (str): The user's original query
        model (str): The model to use for intent classification

    Returns:
        QueryIntent: A dictionary containing the classified intent and confidence score
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    system_prompt = f"""\
{system_date_prompt}
You are an AI assistant skilled in query intent classification. Your task is to analyze a user's query and classify its intent into one of the following categories:
- Informational: Seeking knowledge or answers (e.g., "What is the capital of France?").
- Navigational: Looking for a specific website or resource (e.g., "Facebook login").
- Transactional: Aiming to perform an action like purchasing or downloading (e.g., "Buy iPhone 13").
- Commercial Investigation: Researching before a purchase (e.g., "Best laptops 2023").
Return the intent and a confidence score (0.0 to 1.0) based on the query's clarity and context. Respond only with a JSON object containing 'intent' and 'confidence', without additional text or explanations.
"""

    user_prompt = f"""\
Analyze the following query and classify its intent as informational, navigational, transactional, or commercial_investigation. Provide a confidence score (0.0 to 1.0).

Query: {original_query}

Respond with a JSON object in this format:
{{
    "intent": "<classified_intent>",
    "confidence": <confidence_score>
}}
"""

    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.0,
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content

    import json
    try:
        result = json.loads(response)
        return {
            "intent": result["intent"],
            "confidence": float(result["confidence"])
        }
    except (json.JSONDecodeError, KeyError):
        return {
            "intent": "informational",
            "confidence": 0.5
        }
