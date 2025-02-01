from typing import Optional
from jet.code.markdown_code_extractor import MarkdownCodeExtractor
from jet.llm.main.generation import call_ollama_chat
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
import os
from jet.validation.graph_validation import extract_cypher_block_content
from langchain_community.graphs import MemgraphGraph
from langchain_core.prompts import PromptTemplate
from jet.transformers import format_json
from jet.memory.config import (
    CYPHER_GENERATION_PROMPT,
    CONTEXT_QA_PROMPT,
    CONTEXT_PROMPT_TEMPLATE,
)

initialize_ollama_settings()

# Setup LLM settings
MODEL = "llama3.2"

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


def generate_cypher_query(query: str, graph: MemgraphGraph, *, samples: Optional[str]) -> list[str]:
    prompt = CYPHER_GENERATION_PROMPT.format(
        schema=graph.get_schema,
        # samples=samples,
        prompt=query,
    )
    generated_cypher = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=MODEL,
        options={"seed": 42, "temperature": 0,
                 "num_keep": 0, "num_predict": -1},
    ):
        generated_cypher += chunk

    extractor = MarkdownCodeExtractor()
    results = extractor.extract_code_blocks(generated_cypher)
    transformed_results = [item['code'] for item in results if item['code']]

    return transformed_results


def generate_query(query: str, cypher: str, graph_result_context: str, *, model=MODEL, context: str = "") -> str:
    custom_context = context

    context = CONTEXT_PROMPT_TEMPLATE.format(
        cypher_query_str=cypher.strip('"'),
        graph_result_str=graph_result_context,
    )
    context = custom_context + "\n\n" + context
    context = context.strip()

    prompt = CONTEXT_QA_PROMPT.format(context=context, question=query)
    result = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=model,
        options={"seed": 42, "temperature": 0,
                 "num_keep": 0, "num_predict": -1},
    ):
        result += chunk
    return result


def generate_query_samples(query: str, cypher: str, graph_result_context: str, *, model=MODEL) -> str:
    context = CONTEXT_PROMPT_TEMPLATE.format(
        cypher_query_str=cypher.strip('"'),
        graph_result_str=graph_result_context,
    )
    prompt = CONTEXT_QA_PROMPT.format(context=context, question=query)
    result = ""
    for chunk in call_ollama_chat(
        prompt,
        stream=True,
        model=model,
        options={"seed": 42, "temperature": 0,
                 "num_keep": 0, "num_predict": -1},
    ):
        result += chunk
    return result
