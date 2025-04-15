from jet.file.utils import save_file
from jet.wordnet.words import get_words
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from pydantic import BaseModel, Field
from jet.logger import logger
from jet.token.token_utils import get_model_max_tokens, token_counter
from jet.utils.commands import copy_to_clipboard
from llama_index.core import SimpleDirectoryReader
from jet.vectors.reranker.helpers.structured_llm_rerank import StructuredLLMRerank
from jet.llm.ollama.base import Ollama, OllamaEmbedding, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

# Constants
LLM_MODEL = "llama3.2"
LLM_MAX_TOKENS = get_model_max_tokens(LLM_MODEL)
EMBED_MODEL = "mxbai-embed-large"
EMBED_MAX_TOKENS = get_model_max_tokens(EMBED_MODEL)
CACHE_FILE = "nodes_lyft_2021.pkl"
CHUNK_OVERLAP = 40
CHUNK_SIZE = 256

# Initialize models
llm = Ollama(temperature=0, model=LLM_MODEL,
             request_timeout=300.0, context_window=LLM_MAX_TOKENS)
embed_model = OllamaEmbedding(model_name=EMBED_MODEL)


class DocumentWithRelevance(BaseModel):
    """Document rankings as selected by model."""

    document_number: int = Field(
        description="The number of the document within the provided list"
    )
    relevance: int = Field(
        description="Relevance score from 1-10 of the document to the given query - based on the document content",
        json_schema_extra={"minimum": 1, "maximum": 10},
    )
    feedback: str = Field(
        description="Brief feedback on the document's relevance.",
    )


class DocumentRelevanceList(BaseModel):
    """List of documents with relevance scores."""

    documents: list[DocumentWithRelevance] = Field(
        description="List of documents with relevance scores"
    )
    feedback: str = Field(
        description="Overall feedback on the relevance of all documents.",
    )


def load_documents(data_file: str) -> list:
    """Load documents from cache or read from file if cache is empty."""

    documents = SimpleDirectoryReader(input_files=[data_file]).load_data()

    return documents


def filter_documents(documents: list, keyword: str) -> list:
    """Filter documents containing the given keyword (case-insensitive)."""
    return [doc for doc in documents if keyword.lower() in doc.text.lower()]


def create_index(documents: list) -> VectorStoreIndex:
    """Create a vector store index from given documents."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, tokenizer=get_words)
    nodes = splitter.get_nodes_from_documents(documents=documents)

    return VectorStoreIndex(nodes, show_progress=True, embed_model=embed_model)


def get_retrieved_nodes(
    query_str: str, index: VectorStoreIndex, vector_top_k=10, reranker_top_n=5, with_reranker=False
) -> list[NodeWithScore]:
    """Retrieve and optionally rerank nodes based on the query."""
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(
        index=index, similarity_top_k=vector_top_k)
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        node_token_counts = token_counter(
            [node.text for node in retrieved_nodes], model=LLM_MODEL, prevent_total=True
        )
        max_node_token_count = max(node_token_counts)
        choice_batch_size = int(LLM_MAX_TOKENS * 0.75 / max_node_token_count)

        reranker = StructuredLLMRerank(
            llm=llm,
            choice_batch_size=choice_batch_size,
            top_n=reranker_top_n,
            document_relevance_list_cls=DocumentRelevanceList
        )
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle)

    return retrieved_nodes


def visualize_retrieved_nodes(nodes: list[NodeWithScore]) -> list[dict]:
    """Visualize and copy retrieved nodes to clipboard."""
    results = [{"score": node.score, "text": node.text,
                "feedback": node.metadata.get("feedback", "")} for node in nodes]
    copy_to_clipboard(results)
    logger.pretty(results)
    return results


def save_results(query: str, results: list[dict], output_file: str):
    """Save query results to a JSON file."""
    save_file({"query": query, "results": results}, output_file)
    logger.info("\n\n[DONE]", bright=True)


# Example Execution
if __name__ == "__main__":
    DATA_FILE = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/examples/data/10k/lyft_2021.pdf"
    OUTPUT_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/converted_doc_scripts/llama_index/node_postprocessor/generated/structured_llm_reranker_results.json"

    docs = load_documents(DATA_FILE, CACHE_FILE)
    index = create_index(docs)

    query = "What initiatives are the company focusing on independently of COVID-19?"
    reranked_nodes = get_retrieved_nodes(
        query, index, vector_top_k=40, reranker_top_n=10, with_reranker=True)

    results = visualize_retrieved_nodes(reranked_nodes)
    save_results(query, results, OUTPUT_FILE)
