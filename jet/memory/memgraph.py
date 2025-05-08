from jet.memory.httpx import HttpxClient
from jet.memory.memgraph_types import GraphResponseData, GraphQueryRequest, AuthResponseData, LoginRequest
from jet.transformers.object import make_serializable
from pydantic import BaseModel
from fastapi import HTTPException
import httpx
from typing import TypedDict
from typing import Generator, Optional, Union
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.llm.llm_types import OllamaChatOptions, OllamaChatResponse
from jet.actions.generation import call_ollama_chat
from jet.logger import logger
from jet.llm.ollama.base import initialize_ollama_settings
import os
from jet.validation.graph_validation import extract_cypher_block_content
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from jet.transformers.formatters import format_json
from jet.memory.config import (
    CYPHER_GENERATION_PROMPT,
    CONTEXT_QA_PROMPT,
)

initialize_ollama_settings()

# Setup LLM settings
MODEL = "llama3.2"
# CYPHER_SYSTEM_MESSAGE = """
# You are an AI assistant that follows instructions. You generate cypher queries based on provided context and schema information.
# """.strip()
CYPHER_SYSTEM_MESSAGE = ""

# Setup Memgraph variables
URL = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")


def initialize_graph(url: str, username: str, password: str, data_query: Optional[str] = None) -> MemgraphGraph:
    graph = MemgraphGraph(url=url, username=username,
                          password=password, refresh_schema=False)
    if data_query:
        graph.query("STORAGE MODE IN_MEMORY_ANALYTICAL")
        graph.query("DROP GRAPH")
        graph.query("STORAGE MODE IN_MEMORY_TRANSACTIONAL")

        graph.query(data_query)

    graph.refresh_schema()

    return graph


def generate_cypher_query(query: str, graph: MemgraphGraph, tone_name: str = "an individual", *, num_of_queries=5, samples: Optional[str] = None) -> list[str]:
    prompt = CYPHER_GENERATION_PROMPT.format(
        # schema=graph.get_schema,
        schema=graph.get_structured_schema,
        # samples=samples,
        prompt=query,
        num_of_queries=num_of_queries,
        tone_name=tone_name
    )
    response = generate_query(
        prompt,
        model=MODEL,
        system=CYPHER_SYSTEM_MESSAGE,
    )

    generated_cypher = ""
    for chunk in response:
        generated_cypher += chunk

    extractor = MarkdownCodeExtractor()
    results = extractor.extract_code_blocks(generated_cypher)
    transformed_results = [item['code'] for item in results if item['code']]

    return transformed_results


def generate_query(query: str, tone_name: str = "an individual", *, context: str = "", model=MODEL, stream=False, options: OllamaChatOptions = {}, **kwargs) -> Generator[str | OllamaChatResponse, None, None]:
    prompt = CONTEXT_QA_PROMPT.format(
        context=context, question=query, tone_name=tone_name)

    options = {
        "stream": stream,
        "options": {
            "seed": 0,
            "temperature": 0,
            "num_keep": 0,
            "num_predict": -1,
            **options,
        },
        **kwargs,
    }

    response = call_ollama_chat(prompt, model, **options)

    if stream:
        for chunk in response:
            yield chunk
    else:
        result = response['message']['content']
        yield result


def generate_cypher_context(query: str, graph: MemgraphGraph, tone_name: str, *, num_of_queries: int = 3, top_k: Optional[int] = None) -> str:
    # Generate cypher query
    generated_cypher_queries = generate_cypher_query(
        query, graph, tone_name, num_of_queries=num_of_queries)

    used_cypher_queries = []
    graph_result_contexts = []
    for idx, cypher_query in enumerate(generated_cypher_queries):
        graph_result = graph.query(cypher_query)[:top_k]

        if graph_result:
            logger.newline()
            logger.info(f"Graph Result {idx + 1}:")
            logger.success(graph_result)

            used_cypher_queries.append(cypher_query)
            graph_result_contexts.append(json.dumps(graph_result))

    # Generate query results
    db_results = []
    for item, result in zip(used_cypher_queries, graph_result_contexts):
        db_results.append(f"Query: {item}\nResult: {result}")

    db_results_str = CONTEXT_DB_TEMPLATE.format(
        db_results_str="\n\n".join(db_results))

    schema_str = CONTEXT_SCHEMA_TEMPLATE.format(
        schema_str=graph.get_schema)

    contexts = [
        db_results_str,
        schema_str
    ]
    context = "\n\n".join(contexts)


def authenticate_user(request_data: LoginRequest) -> AuthResponseData:
    url = "http://localhost:3001/auth/login"
    try:
        client = HttpxClient()
        response = client.post(url, json=request_data)
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {
                     e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code, detail=e.response.text
        ) from e  # Raising the original error as context
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e  # Re-raise the original exception


def query_memgraph(request_data: GraphQueryRequest) -> GraphResponseData:
    url = "http://localhost:3001/api/queries"
    try:
        client = HttpxClient()
        response = client.post(url, json=request_data)
        response.raise_for_status()
        return response.json()["data"]
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {
                     e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code, detail=e.response.text
        ) from e  # Raising the original error as context
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e  # Re-raise the original exception


def refresh_auth_token():
    request_data = LoginRequest()
    # Authenticate and get token
    auth_response_data = authenticate_user(request_data)
    new_token = auth_response_data['token']
    return new_token
