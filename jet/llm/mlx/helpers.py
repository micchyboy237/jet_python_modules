from datetime import datetime
from typing import List
from jet.llm.mlx.base import MLX
from jet.llm.mlx.mlx_types import LLMModelType
from jet.logger import logger


def get_system_date_prompt():
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}"


def rewrite_query(original_query: str, model: LLMModelType = "qwen3-1.7b-4bit") -> str:
    """
    Rewrites a query to make it more specific and detailed for better retrieval, while treating years, titles, or events as flexible inputs.

    Args:
        original_query (str): The original user query
        model (str): The model to use for query rewriting

    Returns:
        str: The rewritten query
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    # Updated system prompt to enforce flexibility in years, titles, or events
    system_prompt = f"""
    {system_date_prompt}
    You are an AI assistant specialized in improving search queries. Your task is to rewrite user queries to be more specific and detailed, while ensuring flexibility in years, titles, or events to handle ambiguity and incomplete data gracefully. For example:
    - Instead of "in 2025," use phrases like "around 2025" or "from 2025 or close."
    - Instead of exact titles, use broader terms like "similar to [title]" or "in the style of [title]."
    - For events, use "recent" or "notable" to allow for approximate matches.
    The rewritten query should retrieve relevant information even if exact details are uncertain.
    """

    # Define the user prompt with the original query to be rewritten
    user_prompt = f"""
    Rewrite the following query to make it more specific and detailed, while keeping years, titles, or events flexible to handle ambiguity.
    
    Original query: {original_query}
    
    Rewritten query:
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
    return response


def generate_step_back_query(original_query: str, model: LLMModelType = "qwen3-1.7b-4bit") -> str:
    """
    Generates a more general 'step-back' query to retrieve broader context, with flexible handling of years, titles, or events.

    Args:
        original_query (str): The original user query
        model (str): The model to use for step-back query generation

    Returns:
        str: The step-back query
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    # Updated system prompt to enforce flexibility in years, titles, or events
    system_prompt = f"""
    {system_date_prompt}
    You are an AI assistant specialized in search strategies. Your task is to generate broader, more general versions of specific queries to retrieve relevant background information. Ensure that years, titles, or events are treated as flexible inputs to handle ambiguity and incomplete data gracefully. For example:
    - Use "around [year]" or "recent" instead of specific years.
    - Use general terms like "similar to [title]" or "in the genre of [title]" instead of exact titles.
    - For events, use "notable" or "recent" to allow approximate matches.
    The step-back query should provide useful context even if exact details are uncertain.
    """

    # Define the user prompt with the original query to be generalized
    user_prompt = f"""
    Generate a broader, more general version of the following query that could help retrieve useful background information, keeping years, titles, or events flexible.
    
    Original query: {original_query}
    
    Step-back query:
    """

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
    Decomposes a complex query into simpler sub-queries, with flexible handling of years, titles, or events.

    Args:
        original_query (str): The original complex query
        num_subqueries (int): Number of sub-queries to generate
        model (str): The model to use for query decomposition

    Returns:
        List[str]: A list of simpler sub-queries
    """
    mlx = MLX(model)

    system_date_prompt = get_system_date_prompt()

    # Updated system prompt to enforce flexibility in years, titles, or events
    system_prompt = f"""
    {system_date_prompt}
    You are an AI assistant specialized in breaking down complex questions. Your task is to decompose complex queries into simpler sub-questions that, when answered together, address the original query. Ensure that years, titles, or events in the sub-queries are treated as flexible inputs to handle ambiguity and incomplete data gracefully. For example:
    - Use "around [year]" or "recent" instead of specific years.
    - Use general terms like "similar to [title]" or "in the style of [title]" instead of exact titles.
    - For events, use "notable" or "recent" to allow approximate matches.
    Each sub-query should be clear and focus on a different aspect of the original question, while remaining flexible to uncertainty in details.
    """

    # Define the user prompt with the original query to be decomposed
    user_prompt = f"""
    Break down the following complex query into {num_subqueries} simpler sub-queries. Each sub-query should focus on a different aspect of the original question and treat years, titles, or events as flexible inputs.
    
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
