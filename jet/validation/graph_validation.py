import random
from typing import Optional
from jet.llm import call_ollama_chat
from jet.logger import logger
from jet.validation.cypher_graph_validator import validate_query
from langchain_community.graphs import MemgraphGraph

MODEL = "llama3.2"
PROMPT_TEMPLATE = """
Cypher Query:
[query]
Schema:
[schema]
Validation errors:
[errors]
Fix the Cypher query based on errors:
"""

SYSTEM = f"""
System:
You are a Cypher query corrector. Analyze the provided Cypher query, schema and validation errors, then provide a corrected query that is valid.
Surround the generated query with Cypher block ```cypher\n<generated_query>\n```.
Generated response should only have one Cypher block.
Do not include any other text in the response except the Cypher block.
Data guidelines:
- Follow validation errors to correct the query
- Ensure the corrected query has valid syntax
""".strip()


def validate_cypher(query: str, url: Optional[str] = "bolt://localhost:7687", username: Optional[str] = "", password: Optional[str] = "", attempt: int = 1, max_attempts: int = 10, original_query: Optional[str] = None, generated_error: Optional[Exception] = None) -> dict:
    if original_query is None:
        original_query = query

    if attempt > max_attempts:
        logger.error("Max recursive validation attempts reached.")
        return {"data": query, "corrected": False, "is_valid": False}

    logger.info(f"Validation attempt {attempt}")

    graph = MemgraphGraph(url=url, username=username,
                          password=password, refresh_schema=False)

    result = validate_query(query, graph=graph)
    logger.success(f"Valid Cypher query on attempt {attempt}")

    if result["is_valid"]:
        return {"data": result, "corrected": query != original_query, "is_valid": True}

    logger.error(f"Invalid Cypher query on attempt {attempt}")
    error_prompt = generated_error if generated_error else str(
        result["errors"])

    schema = graph.get_schema

    prompt = f"{
        PROMPT_TEMPLATE
        .replace('[query]', query)
        .replace('[schema]', schema)
        .replace('[errors]', str(error_prompt))
    }"
    try:
        output = ""
        response = call_ollama_chat(
            prompt,
            model=MODEL,
            system=SYSTEM,
            stream=True,
            options={
                "seed": random.randint(1, 9999),
                "temperature": 0,
                "num_keep": 0,
                "num_predict": -1,
            },
        )
        for chunk in response:
            output += chunk

        extracted_result = extract_cypher_block_content(output)
        logger.info(f"Extracted Cypher query content:\n{extracted_result}")
        return validate_cypher(extracted_result, url, username, password, attempt + 1, max_attempts, original_query)
    except Exception as generated_error:
        logger.error(f"Failed to decode AI response: {generated_error}")
        return validate_cypher(query, url, username, password, attempt + 1, max_attempts, original_query, generated_error)


def extract_cypher_block_content(text: str) -> str:
    start = text.find("```cypher")
    if start == -1:
        return text

    start += len("```cypher")
    end = text.find("```", start)
    return text[start:end].strip() if end != -1 else text[start:].strip()
