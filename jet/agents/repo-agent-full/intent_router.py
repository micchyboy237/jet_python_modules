"""
Intent Classification Router for the Repo Agent.
Determines the best strategy for handling a user query.
"""

from typing import Literal

from config import models
from jet.adapters.llama_cpp.llm_utils_observed import achat
from utils import logger

IntentType = Literal["FACTUAL", "SUMMARY", "EXPLORATORY", "DEBUGGING"]


class IntentRouter:
    async def classify(self, query: str) -> IntentType:
        """
        Classify the user's query intent using a lightweight LLM call.
        """
        system_prompt = """
You are an intent classifier for a coding assistant. 
Classify the user's query into one of these categories:
- FACTUAL: Specific questions about code logic, functions, or variables.
- SUMMARY: Requests for high-level overviews, features, or project purpose.
- EXPLORATORY: Questions about project structure, file lists, or finding where something is defined.
- DEBUGGING: Questions related to errors, exceptions, or fixing broken code.

Return ONLY the category name.
"""
        try:
            result = await achat(
                prompt_or_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                model=models.LLM_MODEL,
                temperature=0.0,
                max_tokens=10,
                project_name="repo-agent-router",
            )
            intent = result.content.strip().upper()
            # Fallback to FACTUAL if classification fails
            return (
                intent
                if intent in ["FACTUAL", "SUMMARY", "EXPLORATORY", "DEBUGGING"]
                else "FACTUAL"
            )
        except Exception as e:
            logger.warning(f"Intent classification failed, defaulting to FACTUAL: {e}")
            return "FACTUAL"
