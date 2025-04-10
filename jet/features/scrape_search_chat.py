import json
import os
from typing import Generator, List, Dict, Optional, Type

from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.scrapers.crawler.web_crawler import WebCrawler
from jet.scrapers.utils import scrape_urls, search_data
from jet.search.searxng import NoResultsFoundError, SearchResult, search_searxng
from jet.utils.class_utils import class_to_string
from jet.utils.markdown import extract_json_block_content
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.file.utils import load_file, save_file
from jet.scrapers.preprocessor import html_to_markdown
from jet.code.splitter_markdown_utils import count_md_header_contents, get_md_header_contents
from jet.token.token_utils import get_model_max_tokens, group_texts, split_docs, token_counter
from jet.wordnet.similarity import get_query_similarity_scores
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.evaluators.helpers.base import EvaluationResult
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import ChatResponse, Ollama, OllamaEmbedding
from tqdm import tqdm


# --- Constants ---

HEADERS_PROMPT_TEMPLATE = """
Headers:
{headers}

Instruction:
{instruction}
Query: {query}
Answer:
""".strip()


# --- Pydantic Classes ---

class HeadersQueryResponse(BaseModel):
    data: List[str]


class RelevantDocument(BaseModel):
    document_number: int = Field(..., ge=0)
    confidence: int = Field(..., ge=1, le=10)


class DocumentSelectionResult(BaseModel):
    relevant_documents: List[RelevantDocument]
    evaluated_documents: List[int]
    feedback: str


# --- Core Functions ---


def validate_headers(html: str, min_count: int = 5) -> bool:
    md_text = html_to_markdown(html)
    header_count = count_md_header_contents(md_text)
    return header_count >= min_count


def get_docs_from_html(html: str) -> list[Document]:
    md_text = html_to_markdown(html)
    header_contents = get_md_header_contents(md_text)
    docs = [
        Document(
            text=header["content"],
            metadata={
                "doc_index": i,
                "header": header["header"],
                "header_level": header["header_level"],
            }
        )
        for i, header in enumerate(header_contents)
    ]
    return docs


def get_nodes_from_docs(docs: list[Document], embed_models: str | OLLAMA_EMBED_MODELS | list[str | OLLAMA_EMBED_MODELS], chunk_size: Optional[int] = None, chunk_overlap: int = 40) -> tuple[list[TextNode], dict[str, TextNode]]:
    if isinstance(embed_models, str):
        embed_models = [embed_models]
    model = min(embed_models, key=get_model_max_tokens)
    chunk_size = chunk_size or get_model_max_tokens(model)

    nodes = split_docs(docs, model=model, chunk_size=chunk_size,
                       chunk_overlap=chunk_overlap)
    parent_map = {}
    for node in nodes:
        if node.parent_node and not node.parent_node.node_id in parent_map:
            parent_doc = docs[node.parent_node.metadata["doc_index"]]
            parent_node = TextNode(
                node_id=node.parent_node.node_id,
                text=parent_doc.text,
                metadata=node.parent_node.metadata
            )
            parent_map[node.parent_node.node_id] = parent_node

    return nodes, parent_map


def get_nodes_parent_mapping(nodes: list[TextNode], docs: list[Document]) -> dict:
    parent_map = {}
    for node in nodes:
        if node.parent_node and not node.parent_node.node_id in parent_map:
            parent_doc = docs[node.parent_node.metadata["doc_index"]]
            parent_node = TextNode(
                node_id=node.parent_node.node_id,
                text=parent_doc.text,
                metadata=node.parent_node.metadata
            )
            parent_map[node.parent_node.node_id] = parent_node

    return parent_map


def rerank_nodes(query: str, nodes: List[TextNode], embed_models: List[str], parent_map: Dict[str, TextNode] = {}) -> List[NodeWithScore]:
    texts = [n.text for n in nodes]
    node_map = {n.text: n for n in nodes}
    query_scores = get_query_similarity_scores(
        query, texts, model_name=embed_models)

    results = []
    seen_docs = set()
    for text, score in query_scores[0]["results"].items():
        node = node_map[text]
        parent_info = node.relationships.get(NodeRelationship.PARENT)

        parent_text = text  # fallback to child text
        if parent_info and isinstance(parent_info, RelatedNodeInfo):
            parent_node = parent_map.get(parent_info.node_id)
            if parent_node:
                parent_text = parent_node.text

        doc_index = node.metadata["doc_index"]
        if doc_index not in seen_docs:
            seen_docs.add(doc_index)
            results.append(NodeWithScore(node=TextNode(
                text=parent_text, metadata=node.metadata), score=score))

    return results


def strip_left_hashes(text: str) -> str:
    """
    Removes all leading '#' characters from lines that start with '#'.
    Also strips surrounding whitespace from those lines.

    Args:
        text (str): The input multiline string

    Returns:
        str: Modified string with '#' and extra whitespace removed from matching lines
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#'):
            cleaned_lines.append(stripped_line.lstrip('#').strip())
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


PROMPT_TEMPLATE = PromptTemplate("""
--- Documents ---
{headers}
--- End of Documents ---

Instructions:
You are given a set of structured documents. Your task is to extract all answers relevant to the query using only the content within the documents.

- Use the schema shown below to return your result.
- Only return answers found directly in the documents.
- Remove any duplicates.
- Return *only* the final JSON enclosed in a ```json block.

Schema:
{schema}

Query:
{query}

Answer:
""")

INSTRUCTION = """
Extract relevant information from the documents that directly answer the query.

- Use only the content from the documents provided.
- Remove duplicates when found.
- Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:

Schema:
{schema_str}
""".strip()


def run_scrape_search_chat(
    html,
    query: str,
    output_cls: Type[BaseModel],
    instruction: Optional[str] = None,
    prompt_template: PromptTemplate = PROMPT_TEMPLATE,
    llm_model: OLLAMA_MODEL_NAMES = "mistral",
    embed_models: OLLAMA_EMBED_MODELS | list[OLLAMA_EMBED_MODELS] = "paraphrase-multilingual",
):
    instruction = instruction or INSTRUCTION.format(
        schema_str=class_to_string(output_cls))

    if isinstance(embed_models, str):
        embed_models = [embed_models]

    embed_model = embed_models[0]
    sub_chunk_size = 128
    sub_chunk_overlap = 40

    header_docs = get_docs_from_html(html)
    embed_model = OllamaEmbedding(model_name=embed_model)
    header_tokens: list[list[int]] = embed_model.encode(
        [d.text for d in header_docs])

    header_nodes: list[TextNode] = []
    for doc_idx, doc in tqdm(enumerate(header_docs), total=len(header_docs)):
        sub_nodes = split_docs(
            doc, llm_model, tokens=header_tokens[doc_idx], chunk_size=sub_chunk_size, chunk_overlap=sub_chunk_overlap)
        parent_map = get_nodes_parent_mapping(sub_nodes, header_docs)

        sub_query = f"Query: {query}\n{doc.metadata["header"]}"
        reranked_sub_nodes = rerank_nodes(
            sub_query, sub_nodes, embed_models, parent_map)

        reranked_sub_text = "\n".join([n.text for n in reranked_sub_nodes[:3]])
        reranked_sub_text = reranked_sub_text.lstrip(
            doc.metadata["header"]).strip()
        reranked_sub_text = strip_left_hashes(reranked_sub_text)

        top_sub_node = reranked_sub_nodes[0]
        header_nodes.append(
            TextNode(
                text=f"Document number: {top_sub_node.metadata["doc_index"] + 1}\n```text\n{top_sub_node.metadata["header"]}\n{reranked_sub_text}\n```",
                metadata=top_sub_node.metadata
            )
        )

    header_parent_map = get_nodes_parent_mapping(header_nodes, header_docs)
    reranked_header_nodes = rerank_nodes(
        query, header_nodes, embed_models, header_parent_map)
    # Sort the reranked_header_nodes by metadata['doc_index']
    reranked_header_nodes = sorted(
        reranked_header_nodes, key=lambda node: node.metadata['doc_index'])
    reranked_header_texts = [node.text for node in reranked_header_nodes]

    grouped_header_texts = group_texts(reranked_header_texts, llm_model)

    for idx, header_texts in enumerate(grouped_header_texts):
        headers = "\n\n".join(header_texts)
        llm = Ollama(temperature=0.3, model=llm_model)

        # message = prompt_template.format(
        #     headers=headers,
        #     query=query,
        #     instruction=instruction,
        #     schema=class_to_string(output_cls),
        # )

        # response = llm.chat(
        #     prompt_template,
        #     model=llm_model,
        #     template_vars={
        #         "headers": headers,
        #         "instruction": instruction,
        #         "schema": output_cls.model_json_schema(),
        #         "query": query,
        #     }
        # )

        # json_result = extract_json_block_content(str(response))
        # results = json.loads(json_result)

        response = llm.structured_predict(
            output_cls,
            prompt_template,
            model=llm_model,
            headers=headers,
            instruction=instruction,
            schema=output_cls.model_json_schema(),
            query=query,
        )

        yield {
            "group": idx + 1,
            "query": query,
            "context": headers,
            "response": response,
        }


# Example usage
if __name__ == "__main__":
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models = [
        "mxbai-embed-large",
        "paraphrase-multilingual",
        "granite-embedding",
    ]
    eval_model = llm_model
    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/generated/valid-ids-scraper/philippines_national_id_registration_tips_2025/scraped_html.html"
    output_dir = f"/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/{os.path.splitext(os.path.basename(__file__))[0]}"
    query = "What are the steps in registering a National ID in the Philippines?"

    html = load_file(data_file)

    result = run_scrape_search_chat(
        html,
        llm_model,
        embed_models,
        eval_model,
        output_dir,
        query,
    )

    # if result["search_eval"].passing:
    save_file({
        "query": result["query"],
        "results": result["search_nodes"]
    }, os.path.join(output_dir, "top_nodes.json"))

    save_file(result["search_eval"], os.path.join(
        output_dir, "eval_context_relevancy.json"))

    history = "\n\n".join([
        f"## Query\n\n{result["query"]}",
        f"## Context\n\n{result["context"]}",
        f"## Response\n\n{result["response"]}",
    ])
    save_file(history, os.path.join(output_dir, "llm_chat_history.md"))
