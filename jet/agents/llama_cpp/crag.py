import sys

sys.path.append(
    "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques"
)

import json
from typing import List, Tuple, Union

from dotenv import load_dotenv
from evaluation.evalute_rag import *
from helper_functions import replace_t_with_space
from jet.adapters.langchain.factory import get_chat_openai, get_openai_embeddings
from jet.adapters.langchain.tools.searxng_search_tool import SearXNGSearchResults
from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL
from jet.logger import logger
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts.prompt import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv()


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """
    loader = PyPDFLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    embeddings = get_openai_embeddings(EMBED_MODEL_LG)
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    return vectorstore


path = "/Users/jethroestrada/Desktop/External_Projects/AI/examples/RAG_Techniques/data/Understanding_Climate_Change.pdf"
vectorstore = encode_pdf(path)

llm = get_chat_openai(model=LLM_MODEL, max_tokens=1000, temperature=0)

search = SearXNGSearchResults()


class RetrievalEvaluatorInput(BaseModel):
    """A relevance score between 0 and 1 indicating document query match."""

    model_config = ConfigDict(populate_by_name=True)

    relevance_score: float = Field(
        ...,
        alias="relevance",
        description="The relevance score of the document to the query. Must be a float between 0.0 and 1.0.",
    )


def retrieval_evaluator(query: str, document: str) -> float:
    prompt = PromptTemplate(
        input_variables=["query", "document"],
        template=(
            "On a scale from 0 to 1, how relevant is the following document to the query?\n\n"
            "Query: {query}\nDocument: {document}\n\n"
            "Respond ONLY with valid JSON in this EXACT format:\n"
            '{{"relevance_score": 0.0}}\n'
            "Do NOT include any other fields like 'reasoning'. Do NOT add explanations."
        ),
    )
    chain = prompt | llm.with_structured_output(
        RetrievalEvaluatorInput, method="json_mode"
    )
    input_variables = {"query": query, "document": document}
    result = chain.invoke(input_variables).relevance_score
    return result


class KnowledgeRefinementInput(BaseModel):
    """Key points extracted from a document as bullet points."""

    key_points: Union[str, List[str]] = Field(
        ..., description="The extracted key information formatted as bullet points."
    )

    @field_validator("key_points", mode="before")
    @classmethod
    def normalize_key_points(cls, v: Union[str, List[str]]) -> str:
        """Accept both str and list from LLM; always return a string."""
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return str(v)


def knowledge_refinement(document: str, query: str = "") -> List[str]:
    prompt = PromptTemplate(
        input_variables=["document", "query"],
        template=(
            "Extract the key information from the following document that answers the query.\n\n"
            "Query: {query}\n"
            "Document: {document}\n\n"
            "Respond ONLY with valid JSON in this EXACT format:\n"
            '{{"key_points": "- point 1\\n- point 2"}}\n'
            "Do NOT include any other fields. Do NOT add explanations."
        ),
    )
    chain = prompt | llm.with_structured_output(
        KnowledgeRefinementInput, method="json_mode"
    )
    input_variables = {"document": document, "query": query}
    result = chain.invoke(input_variables).key_points
    return [point.strip() for point in result.split("\n") if point.strip()]


class QueryRewriterInput(BaseModel):
    """A rewritten query optimized for web search."""

    query: str = Field(..., description="The rewritten search query.")


def rewrite_query(query: str) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Rewrite the following query to make it more suitable for a web search.\n\n"
            "Query: {query}\n\n"
            "Respond ONLY with valid JSON in this EXACT format:\n"
            '{{"query": "rewritten query here"}}\n'
            "Do NOT include any other fields. Do NOT add explanations."
        ),
    )
    chain = prompt | llm.with_structured_output(QueryRewriterInput, method="json_mode")
    input_variables = {"query": query}
    return chain.invoke(input_variables).query.strip()


def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """
    Parse search results into a list of title-link tuples.
    Handles both JSON and text-based formats.

    Args:
        results_string (str): A JSON-formatted string or text-formatted search results.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the title and link of a search result.
                               If parsing fails, an empty list is returned.
    """
    try:
        # Try parsing as JSON first
        results = json.loads(results_string)
        return [
            (result.get("title", "Untitled"), result.get("url", result.get("link", "")))
            for result in results
        ]
    except json.JSONDecodeError:
        # If not JSON, try parsing the text format with url/title/content keys
        try:
            # Split by double newlines to get individual results
            entries = results_string.strip().split("\n\n")
            parsed = []
            for entry in entries:
                lines = entry.split("\n")
                entry_dict = {}
                for line in lines:
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        entry_dict[key] = value
                title = entry_dict.get("title", "Untitled")
                url = entry_dict.get("url", "")
                parsed.append((title, url))
            return parsed if parsed else []
        except Exception as e:
            logger.warning(f"Error parsing search results text: {e}")
            return []


def retrieve_documents(query: str, faiss_index: FAISS, k: int = 3) -> List[str]:
    """
    Retrieve documents based on a query using a FAISS index.

    Args:
        query (str): The query string to search for.
        faiss_index (FAISS): The FAISS index used for similarity search.
        k (int): The number of top documents to retrieve. Defaults to 3.

    Returns:
        List[str]: A list of the retrieved document contents.
    """
    docs = faiss_index.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def evaluate_documents(query: str, documents: List[str]) -> List[float]:
    """
    Evaluate the relevance of documents based on a query.

    Args:
        query (str): The query string.
        documents (List[str]): A list of document contents to evaluate.

    Returns:
        List[float]: A list of relevance scores for each document.
    """
    return [retrieval_evaluator(query, doc) for doc in documents]


def perform_web_search(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Perform a web search based on a query.

    Args:
        query (str): The query string to search for.

    Returns:
        Tuple[str, List[Tuple[str, str]]]:
            - A string of refined knowledge obtained from the web search.
            - A list of tuples containing titles and links of the sources.
    """
    rewritten_query = rewrite_query(query)

    # Handle tuple return from search.run() when response_format is "content_and_artifact"
    result = search.run(rewritten_query)
    if isinstance(result, tuple):
        web_results = result[0]  # Get the formatted string
    else:
        web_results = result

    web_knowledge_list = knowledge_refinement(web_results, rewritten_query)
    sources = parse_search_results(web_results)

    # Join list to string before returning
    web_knowledge = (
        "\n".join(web_knowledge_list)
        if isinstance(web_knowledge_list, list)
        else web_knowledge_list
    )

    return web_knowledge, sources


def generate_response(
    query: str, knowledge: str, sources: List[Tuple[str, str]]
) -> str:
    """
    Generate a response to a query using knowledge and sources.

    Args:
        query (str): The query string.
        knowledge (str): The refined knowledge to use in the response.
        sources (List[Tuple[str, str]]): A list of tuples containing titles and links of the sources.

    Returns:
        str: The generated response.
    """
    response_prompt = PromptTemplate(
        input_variables=["query", "knowledge", "sources"],
        template=(
            "Based on the following knowledge, answer the query. "
            "Include the sources with their links (if available) at the end of your answer:\n"
            "Query: {query}\nKnowledge: {knowledge}\nSources: {sources}\nAnswer:"
        ),
    )
    input_variables = {
        "query": query,
        "knowledge": knowledge,
        "sources": "\n".join(
            [f"{title}: {link}" if link else title for title, link in sources]
        ),
    }
    response_chain = response_prompt | llm
    return response_chain.invoke(input_variables).content


def crag_process(query: str, faiss_index: FAISS) -> str:
    """
    Process a query by retrieving, evaluating, and using documents or performing a web search to generate a response.

    Args:
        query (str): The query string to process.
        faiss_index (FAISS): The FAISS index used for document retrieval.

    Returns:
        str: The generated response based on the query.
    """
    logger.info(f"Processing query: {query}")

    # Retrieve and evaluate documents
    retrieved_docs = retrieve_documents(query, faiss_index)
    eval_scores = evaluate_documents(query, retrieved_docs)
    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    logger.info(f"Evaluation scores: {eval_scores}")

    max_score = max(eval_scores)
    sources = []

    if max_score > 0.7:
        # Correct: Use retrieved document directly
        logger.info("Action: Correct - Using retrieved document")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        final_knowledge = best_doc  # Already a string
        sources.append(("Retrieved document", ""))

    elif max_score < 0.3:
        # Incorrect: Perform web search
        logger.info("Action: Incorrect - Performing web search")
        final_knowledge, sources = perform_web_search(query)
        # Safety check: ensure knowledge is a string
        if isinstance(final_knowledge, list):
            final_knowledge = "\n".join(final_knowledge)

    else:
        # Ambiguous: Combine both
        logger.info("Action: Ambiguous - Combining retrieved document and web search")
        best_doc = retrieved_docs[eval_scores.index(max_score)]
        retrieved_knowledge_list = knowledge_refinement(best_doc, query)
        web_knowledge, web_sources = perform_web_search(query)

        # Join lists to strings
        retrieved_knowledge = (
            "\n".join(retrieved_knowledge_list)
            if isinstance(retrieved_knowledge_list, list)
            else retrieved_knowledge_list
        )
        web_knowledge = (
            "\n".join(web_knowledge)
            if isinstance(web_knowledge, list)
            else web_knowledge
        )

        final_knowledge = "\n".join([retrieved_knowledge, web_knowledge])
        sources = [("Retrieved document", "")] + web_sources

    # Final safety check: ensure knowledge is always a string
    if isinstance(final_knowledge, list):
        final_knowledge = "\n".join(final_knowledge)

    logger.info("Final knowledge:")
    logger.info(final_knowledge)

    logger.info("Sources:")
    for title, link in sources:
        logger.info(f"{title}: {link}" if link else title)

    logger.info("Generating response...")
    response = generate_response(query, final_knowledge, sources)
    logger.success("Response generated")

    return response


if __name__ == "__main__":
    query = "What are the main causes of climate change?"
    result = crag_process(query, vectorstore)
    print(f"Query: {query}")
    print(f"Answer: {result}")

    print("\n" + "=" * 80 + "\n")

    query = "how did harry beat quirrell?"
    result = crag_process(query, vectorstore)
    print(f"Query: {query}")
    print(f"Answer: {result}")
