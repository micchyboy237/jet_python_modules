from datetime import datetime
from typing import List, Tuple, Union
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelType, MLXTokenizer, ModelType
from jet.llm.mlx.models import resolve_model
from jet.logger import logger
import mlx.nn as nn


def load_model(model: ModelType) -> Tuple[nn.Module, MLXTokenizer]:
    from mlx_lm import load

    model_path = str(resolve_model(model))

    return load(model_path)


def get_system_date_prompt():
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"


def rewrite_query(original_query: str, model: LLMModelType = "qwen3-1.7b-4bit") -> str:
    """
    Rewrites a query to be more precise and detailed for improved search results, while allowing flexibility for uncertain years, titles, or events.

    Args:
        original_query (str): The user's original query
        model (str): The model to use for query rewriting

    Returns:
        str: The rewritten query
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    system_prompt = f"""\
{system_date_prompt}
You are an AI assistant designed to enhance search queries. Your goal is to rewrite the user's query to be more specific and detailed, improving retrieval accuracy. To handle ambiguity, treat years, titles, or events flexibly:
- For years, use phrases like "around [year]" or "near [year]" instead of exact years.
- For titles, use broad terms like "works similar to [title]" or "in the style of [title]."
- For events, use terms like "recent" or "notable" to allow approximate matches.
The rewritten query should retrieve relevant results even if the user's input contains uncertain or incomplete details.
Respond only with the rewritten query, without any additional text, labels, or explanations."""

    user_prompt = f"""\
Rewrite the following query to be more specific and detailed, while keeping years, titles, or events flexible to account for ambiguity. Respond only with the rewritten query.

Original query: {original_query}
Rewritten query:"""

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
    return response


def generate_step_back_query(original_query: str, model: LLMModelType = "qwen3-1.7b-4bit") -> str:
    """
    Creates a broader, more general version of a query to provide useful context, with flexible handling of years, titles, or events.

    Args:
        original_query (str): The user's original query
        model (str): The model to use for generating the step-back query

    Returns:
        str: The broader step-back query
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    system_prompt = f"""\
{system_date_prompt}
You are an AI assistant skilled in search optimization. Your task is to create a broader, more general version of the user's query to gather relevant background information. To account for ambiguity, handle years, titles, or events flexibly:
- Use terms like "around [year]" or "recent" instead of specific years.
- Refer to titles with phrases like "works similar to [title]" or "in the genre of [title]."
- For events, use "notable" or "recent" to allow for approximate matches.
The step-back query should provide useful context even if the exact details in the original query are uncertain."""

    user_prompt = f"""\
Create a broader, more general version of the following query to retrieve useful background information, keeping years, titles, or events flexible.

Original query: {original_query}

Step-back query:"""

    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.1,
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
    return response


def decompose_query(original_query: str, num_subqueries: int = 3, model: LLMModelType = "qwen3-1.7b-4bit") -> List[str]:
    """
    Breaks down a complex query into simpler sub-queries, with flexible handling of years, titles, or events.

    Args:
        original_query (str): The user's complex query
        num_subqueries (int): Number of sub-queries to generate
        model (str): The model to use for query decomposition

    Returns:
        List[str]: A list of simpler sub-queries
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    system_prompt = f"""\
{system_date_prompt}
You are an AI assistant skilled in simplifying complex questions. Your task is to break down a complex query into {num_subqueries} simpler sub-queries that together address the original question. To handle ambiguity, treat years, titles, or events flexibly:
- Use terms like "around [year]" or "recent" instead of exact years.
- Refer to titles with phrases like "works similar to [title]" or "in the style of [title]."
- For events, use "notable" or "recent" to allow approximate matches.
Each sub-query should be clear, focus on a distinct aspect of the original query, and remain flexible to uncertainty in details.
"""

    user_prompt = f"""\
Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should address a different aspect of the original question and treat years, titles, or events flexibly.

Original query: {original_query}

Generate {num_subqueries} sub-queries, one per line, in this format:
1. [First sub-query].
2. [Second sub-query].
And so on...
"""

    response = ""
    for chunk in mlx.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.2,
        verbose=True
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
    lines = response.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # Remove the number and leading space
            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)

    return sub_queries


__all__ = [
    "load_model",
    "get_system_date_prompt",
    "rewrite_query",
    "generate_step_back_query",
    "decompose_query",
]
