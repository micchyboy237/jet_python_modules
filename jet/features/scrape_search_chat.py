import multiprocessing
from typing import List, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from typing import Generator, List, Dict, Optional, Type

from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.scrapers.crawler.web_crawler import WebCrawler
from jet.scrapers.utils import safe_path_from_url, scrape_urls, search_data, validate_headers
from jet.search.searxng import NoResultsFoundError, SearchResult, search_searxng
from jet.transformers.formatters import format_json
from jet.utils.class_utils import class_to_string
from jet.utils.markdown import extract_json_block_content
from llama_index.core.prompts.base import PromptTemplate
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.file.utils import load_file, save_file
from jet.scrapers.preprocessor import html_to_markdown
from jet.code.splitter_markdown_utils import count_md_header_contents, get_md_header_contents
from jet.token.token_utils import get_model_max_tokens, group_nodes, group_texts, split_docs, token_counter
from jet.wordnet.similarity import get_query_similarity_scores
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.evaluators.helpers.base import EvaluationResult
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import ChatResponse, Ollama, OllamaEmbedding
from tqdm import tqdm

# Access the wn object before you enter into your threading
# https://stackoverflow.com/questions/27433370/what-would-cause-wordnetcorpusreader-to-have-no-attribute-lazycorpusloader
# Fixes AttributeError: 'WordListCorpusReader' object has no attribute '_LazyCorpusLoader__args'
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import WordNetError
import sys
import time
import threading
cachedStopWords = stopwords.words("english")

max_workers = multiprocessing.cpu_count() // 2


# --- Constants ---

PROMPT_TEMPLATE = PromptTemplate("""
--- Documents ---
{headers}
--- End of Documents ---

Schema:
{schema}

Instructions:
{instruction}

Query:
{query}

Answer:
""")

INSTRUCTION = """
Your task is to extract and return all information that directly answers the query from the provided documents.

- Only use the information present in the documents. Do not include any external or prior knowledge.
- Your output must strictly match the schema format provided below.
- Use the field descriptions in the schema to guide what content belongs in each field.
- Do not infer or fabricate data for fields not clearly stated in the documents. Leave optional fields null if information is missing.
- Remove duplicate or redundant results.
- Return only a single JSON object conforming to the schema. Do not include explanations, commentary, or additional text.

Ensure the structure and field names exactly match the schema.
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


def process_document(
    doc_idx: int,
    doc,
    header_docs: list[Document],
    header_tokens: List[List[int]],
    query: str,
    llm_model: str,
    embed_models: list,
    sub_chunk_size: int,
    sub_chunk_overlap: int,
) -> TextNode:
    sub_nodes = split_docs(
        doc,
        llm_model,
        tokens=header_tokens[doc_idx],
        chunk_size=sub_chunk_size,
        chunk_overlap=sub_chunk_overlap,
    )
    parent_map = get_nodes_parent_mapping(sub_nodes, header_docs)

    sub_query = f"Query: {query}\n{doc.metadata['header']}"
    reranked_sub_nodes = rerank_nodes(
        sub_query, sub_nodes, embed_models, parent_map)

    reranked_sub_text = "\n".join([n.text for n in reranked_sub_nodes[:3]])
    reranked_sub_text = reranked_sub_text.lstrip(
        doc.metadata['header']).strip()
    reranked_sub_text = strip_left_hashes(reranked_sub_text)

    top_sub_node = reranked_sub_nodes[0]
    return TextNode(
        text=(
            f"Document number: {top_sub_node.metadata['doc_index'] + 1}\n"
            f"```text\n{top_sub_node.metadata['header']}\n{reranked_sub_text}\n```"
        ),
        metadata=top_sub_node.metadata,
    )


def _process_document_star(args):
    return process_document(*args)


def run_scrape_search_chat(
    html,
    query: str,
    output_cls: Type[BaseModel],
    instruction: Optional[str] = None,
    prompt_template: PromptTemplate = PROMPT_TEMPLATE,
    llm_model: OLLAMA_MODEL_NAMES = "mistral",
    embed_models: OLLAMA_EMBED_MODELS | list[OLLAMA_EMBED_MODELS] = "paraphrase-multilingual",
    min_tokens_per_group: Optional[int | float] = None,
    max_tokens_per_group: Optional[int | float] = None,
):
    model_max_tokens = get_model_max_tokens(llm_model)
    if not max_tokens_per_group:
        max_tokens_per_group = model_max_tokens * 0.5
    if not min_tokens_per_group:
        min_tokens_per_group = max_tokens_per_group * 0.5

    instruction = instruction or INSTRUCTION

    if isinstance(embed_models, str):
        embed_models = [embed_models]

    embed_model = embed_models[0]
    sub_chunk_size = 128
    sub_chunk_overlap = 40

    header_docs = get_docs_from_html(html)
    shared_header_doc = header_docs[0]

    embed_model = OllamaEmbedding(model_name=embed_model)
    header_tokens: list[list[int]] = embed_model.encode(
        [d.text for d in header_docs])

    process_args = [
        (doc_idx, doc, header_docs, header_tokens, query, llm_model,
         embed_models, sub_chunk_size, sub_chunk_overlap)
        for doc_idx, doc in enumerate(header_docs)
    ]

    header_nodes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_document_star, args)
                   for args in process_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            header_nodes.append(future.result())

    header_parent_map = get_nodes_parent_mapping(header_nodes, header_docs)

    # Remove first h1
    filtered_header_nodes = [
        node for node in header_nodes if node.metadata["doc_index"] != shared_header_doc.metadata["doc_index"]]
    # Rerank headers
    reranked_header_nodes = rerank_nodes(
        query, filtered_header_nodes, embed_models, header_parent_map)
    # Sort reranked results by doc index
    sorted_header_nodes = sorted(
        reranked_header_nodes, key=lambda node: node.metadata['doc_index'])

    grouped_header_nodes = group_nodes(
        sorted_header_nodes, llm_model, max_tokens=max_tokens_per_group)

    for idx, header_nodes in enumerate(grouped_header_nodes):
        header_texts = [node.text for node in header_nodes]
        # Prepend shared context
        header_texts = [shared_header_doc.text] + header_texts
        headers = "\n\n".join(header_texts)
        header_tokens: int = token_counter(headers, llm_model)
        llm = Ollama(temperature=0.3, model=llm_model)

        response = llm.structured_predict(
            output_cls,
            prompt_template,
            model=llm_model,
            headers=headers,
            instruction=instruction,
            query=query,
            schema=json.dumps(output_cls.model_json_schema(), indent=2),
        )
        response_tokens: int = token_counter(
            format_json(response.results), llm_model)
        reranked_nodes = [
            {
                "doc": node.metadata["doc_index"] + 1,
                "score": node.score,
                "text": node.text,
                "metadata": node.metadata,
            }
            for node in reranked_header_nodes
        ]
        yield {
            "group": idx + 1,
            "query": query,
            "context": headers,
            "context_tokens": header_tokens,
            "context_nodes": reranked_nodes,
            "response": response,
            "response_tokens": response_tokens,
        }


# Example usage
if __name__ == "__main__":
    class Answer(BaseModel):
        title: str = Field(
            ..., description="The exact title of the anime, as it appears in the document.")
        document_number: int = Field(
            ..., description="The number of the document that includes this anime (e.g., 'Document number: 3').")
        release_year: Optional[int] = Field(
            description="The most recent known release year of the anime, if specified in the document.")

    class QueryResponse(BaseModel):
        results: list[Answer] = Field(
            default_factory=list,
            description="List of relevant anime titles extracted from the documents, matching the user's query. Each entry includes the title, source document number, and release year (if known)."
        )

    output_cls = QueryResponse

    # --- Inputs ---
    # llm_model = "gemma3:4b"
    llm_model = "mistral"
    embed_models: list[OLLAMA_EMBED_MODELS] = [
        "paraphrase-multilingual",
        # "mxbai-embed-large",
    ]
    eval_model = llm_model
    output_dir = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    query = "Top otome villainess anime 2025"
    # query = construct_browser_query(
    #     search_terms="top 10 romantic comedy anime",
    #     include_sites=["myanimelist.net",
    #                    "anilist.co", "animenewsnetwork.com"],
    #     exclude_sites=["wikipedia.org", "imdb.com"],
    #     # after_date="2024-01-01",
    #     # before_date="2025-04-05"
    # )
    min_header_count = 5

    # Search urls
    search_results = search_data(query)
    urls = [item["url"] for item in search_results]

    scraped_urls_results = scrape_urls(urls)
    pbar = tqdm(total=len(urls))
    for url, html in scraped_urls_results:
        pbar.set_description(f"URL: {url}")

        if not validate_headers(html, min_count=min_header_count):
            logger.warning(
                f"Skipping url: {url} due to header count < {min_header_count}")
            continue

        logger.info(f"Scraping url: {url}")
        sub_dir = safe_path_from_url(url, output_dir)

        html_file = f"{sub_dir}/scraped_html.html"
        save_file(html, html_file)

        response_generator = run_scrape_search_chat(
            html=html,
            query=query,
            output_cls=output_cls,
            llm_model=llm_model,
            embed_models=embed_models,
        )

        class ContextNodes(TypedDict):
            group: int
            tokens: int
            nodes: list[NodeWithScore]

        contexts: list[str] = []

        context_nodes: list[ContextNodes] = []
        context_nodes_dict = {
            "query": query,
            "data": context_nodes,
        }

        class Results(TypedDict):
            group: int
            tokens: int
            results: list[Answer]

        results: list[Results] = []
        results_dict = {
            "query": query,
            "data": results
        }
        for response in response_generator:
            group = response["group"]

            context_tokens = response["context_tokens"]
            context: str = response["context"]
            context_nodes.append(
                {"group": group, "tokens": context_tokens, "nodes": response["context_nodes"]})

            response_obj: QueryResponse = response["response"]
            response_tokens = response["response_tokens"]
            results.append(
                {"group": group, "tokens": response_tokens, "results": response_obj.results})

            contexts.append(f"<!-- Group {group} -->\n\n{context}")
            save_file("\n\n".join(contexts),
                      os.path.join(output_dir, f"context_nodes.md"))
            save_file(context_nodes_dict, os.path.join(
                output_dir, f"context_nodes.json"))
            save_file(results_dict, f"{output_dir}/results.json")

        pbar.update(1)
