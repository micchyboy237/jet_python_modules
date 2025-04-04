import os
from typing import Generator, List, Dict, Optional

from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.scrapers.crawler.web_crawler import WebCrawler
from jet.scrapers.utils import scrape_urls, search_data
from jet.search.searxng import NoResultsFoundError, SearchResult, search_searxng
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.file.utils import load_file, save_file
from jet.scrapers.preprocessor import html_to_markdown
from jet.code.splitter_markdown_utils import count_md_header_contents, get_md_header_contents
from jet.token.token_utils import get_model_max_tokens, split_docs, token_counter
from jet.wordnet.similarity import get_query_similarity_scores
from llama_index.core.schema import Document, NodeRelationship, NodeWithScore, RelatedNodeInfo, TextNode
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.llm.evaluators.helpers.base import EvaluationResult
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.ollama.base import ChatResponse, Ollama, OllamaEmbedding


# --- Pydantic Classes ---


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


def rerank_nodes(query: str, nodes: List[TextNode], embed_models: List[str], parent_map: Dict[str, TextNode]) -> List[NodeWithScore]:
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


def run_scrape_search_chat(
    html,
    llm_model,
    embed_models,
    eval_model,
    output_dir,
    query,
    min_headers: int = 5,
    buffer: Optional[int] = None,
):
    max_model_tokens = get_model_max_tokens(llm_model)
    buffer = buffer or max_model_tokens - int(max_model_tokens * 0.6)
    max_context_tokens = max_model_tokens - buffer

    if not validate_headers(html, min_count=min_headers):
        return None

    # Setup nodes
    docs = get_docs_from_html(html)
    nodes, parent_map = get_nodes_from_docs(docs, embed_models)

    # Search nodes
    reranked_nodes = rerank_nodes(query, nodes, embed_models, parent_map)

    # Filter reranked_nodes to fit in llm chat context
    reranked_nodes_tokens: list[int] = token_counter(
        [node.text for node in reranked_nodes], llm_model, prevent_total=True)
    top_nodes: list[NodeWithScore] = []
    current_top_nodes_tokens: list[int] = []
    for node, tokens in zip(reranked_nodes, reranked_nodes_tokens):
        total_top_nodes_tokens = sum(current_top_nodes_tokens + [tokens])
        if total_top_nodes_tokens < max_context_tokens:
            top_nodes.append(node)
            current_top_nodes_tokens.append(tokens)
        else:
            break

    # Evaluate contexts
    logger.debug(f"Evaluating contexts ({len(current_top_nodes_tokens)})...")
    eval_result = evaluate_context_relevancy(
        eval_model, query, [n.text for n in top_nodes], buffer=buffer)

    if eval_result.passing:
        logger.success(f"Context relevancy passed ({len(top_nodes)})")
    else:
        logger.error(f"Context relevancy failed ({len(top_nodes)})")
        return None

    # Chat LLM
    logger.debug(f"Generating chat response...")
    context = "\n\n".join([n.text for n in top_nodes])
    llm = Ollama(temperature=0.3, model=llm_model, buffer=buffer)
    response = llm.chat(query, context=context)

    return {
        "query": query,
        "context": context,
        "nodes": nodes,
        "parent_nodes": list(parent_map.values()),
        "search_nodes": top_nodes,
        "search_eval": eval_result,
        "response": response,
    }


# Example usage
if __name__ == "__main__":
    llm_model = "gemma3:4b"
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

    if result["search_eval"].passing:
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
