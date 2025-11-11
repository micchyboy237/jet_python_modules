"""
Context-Engineering: RAG Recipes for Retrieval-Augmented Generation
===================================================================

This module demonstrates practical implementations of Retrieval-Augmented
Generation (RAG) patterns for enhancing LLM contexts with external knowledge.
We focus on minimal, efficient implementations that highlight the key concepts
without requiring complex infrastructure.

Key concepts covered:
1. Basic RAG pipeline construction
2. Context window management and chunking strategies 
3. Embedding and retrieval techniques
4. Measuring retrieval quality and relevance
5. Context integration patterns
6. Advanced RAG variations

Usage:
    # In Jupyter or Colab:
    %run 04_rag_recipes.py
    # or
    from rag_recipes import SimpleRAG, ChunkedRAG, HybridRAG
"""

import json
import time
import numpy as np
import tiktoken
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

from jet._token.token_utils import detokenize, token_counter, tokenize
from jet.adapters.llama_cpp.llm import LlamacppLLM
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
import shutil
import pathlib
import logging

BASE_OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. Install with: pip install openai")

try:
    import dotenv
    dotenv.load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    logger.warning("python-dotenv not found. Install with: pip install python-dotenv")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not found. Install with: pip install scikit-learn")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not found. Install with: pip install numpy")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not found. Install with: pip install faiss-cpu or faiss-gpu")

# Constants
DEFAULT_MODEL = "qwen3-instruct-2507:4b"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 3

@dataclass
class Document:
    """Represents a document or chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

    def __post_init__(self):
        """Initialize default values if not provided."""
        if self.metadata is None:
            self.metadata = {}
        if self.id is None:
            import hashlib
            self.id = hashlib.md5(self.content.encode()).hexdigest()[:8]

def setup_client(api_key=None, model=DEFAULT_MODEL):
    client = LlamacppLLM(model=model, verbose=True)
    return client, model

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    return token_counter(text, model=model)

def generate_embedding(
    text: str,
    client=None,
    model: str = DEFAULT_EMBEDDING_MODEL
) -> List[float]:
    embedder = LlamacppEmbedding(model=model)
    embeddings = embedder.encode(text, return_format="list")[0]
    return embeddings

def generate_response(
    prompt: str,
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_message: str = "You are a helpful assistant."
) -> Tuple[str, Dict[str, Any]]:
    if client is None:
        client, model = setup_client(model=model)
        if client is None:
            return "ERROR: No API client available", {"error": "No API client"}

    prompt_tokens = count_tokens(prompt, model)
    system_tokens = count_tokens(system_message, model)

    metadata = {
        "prompt_tokens": prompt_tokens,
        "system_tokens": system_tokens,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": time.time()
    }

    try:
        start_time = time.time()
        response_stream = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        response = ""
        for chunk in response_stream:
            response += chunk
        latency = time.time() - start_time

        response_text = response
        response_tokens = count_tokens(response_text, model)
        metadata.update({
            "latency": latency,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + system_tokens + response_tokens,
            "token_efficiency": response_tokens / (prompt_tokens + system_tokens) if (prompt_tokens + system_tokens) > 0 else 0,
            "tokens_per_second": response_tokens / latency if latency > 0 else 0
        })

        return response_text, metadata

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        metadata["error"] = str(e)
        return f"ERROR: {str(e)}", metadata

def format_metrics(metrics: Dict[str, Any]) -> str:
    key_metrics = {
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "response_tokens": metrics.get("response_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "latency": f"{metrics.get('latency', 0):.2f}s",
        "token_efficiency": f"{metrics.get('token_efficiency', 0):.2f}"
    }
    return " | ".join([f"{k}: {v}" for k, v in key_metrics.items()])

def display_response(
    prompt: str,
    response: str,
    retrieved_context: Optional[str] = None,
    metrics: Dict[str, Any] = None,
    show_prompt: bool = True,
    show_context: bool = True,
    output_dir: Optional[pathlib.Path] = None,
) -> None:
    """
    Save a prompt-response pair with metrics to files under output_dir.
    """
    if output_dir is None:
        output_dir = BASE_OUTPUT_DIR / "example_misc"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if show_prompt:
        (output_dir / "query.md").write_text(f"# Query\n\n```\n{prompt}\n```")
    if retrieved_context and show_context:
        (output_dir / "context.md").write_text(f"# Retrieved Context\n\n```\n{retrieved_context}\n```")
    (output_dir / "response.md").write_text(f"# Response\n\n{response}")
    if metrics:
        metrics_str = format_metrics(metrics)
        (output_dir / "metrics.txt").write_text(metrics_str)

def save_visualization(fig: plt.Figure, output_dir: pathlib.Path, name: str = "plot") -> None:
    """
    Save a matplotlib figure to PNG and SVG under output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight", dpi=150)
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

def text_to_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    model: str = DEFAULT_MODEL
) -> List[Document]:
    if not text:
        return []
    tokens = tokenize(text, model)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_end = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:chunk_end]
        chunk_text = detokenize(chunk_tokens, model)
        chunks.append(Document(
            content=chunk_text,
            metadata={
                "start_idx": i,
                "end_idx": chunk_end,
                "chunk_size": len(chunk_tokens)
            }
        ))
        i += max(1, chunk_size - chunk_overlap)
    return chunks

def extract_document_batch_embeddings(
    documents: List[Document],
    client=None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 10
) -> List[Document]:
    if not documents:
        return []
    texts = [doc.content for doc in documents]
    embedder = LlamacppEmbedding(model=model)
    embeddings_matrix = embedder.encode(texts, return_format="list")
    for doc, embeddings in zip(documents, embeddings_matrix):
        doc.embedding = embeddings
    return documents

def similarity_search(
    query_embedding: List[float],
    documents: List[Document],
    top_k: int = DEFAULT_TOP_K
) -> List[Tuple[Document, float]]:
    if not NUMPY_AVAILABLE:
        logger.error("NumPy required for similarity search")
        return []
    docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
    if not docs_with_embeddings:
        logger.warning("No documents with embeddings found")
        return []
    query_embedding_np = np.array(query_embedding).reshape(1, -1)
    doc_embeddings = np.array([doc.embedding for doc in docs_with_embeddings])
    if SKLEARN_AVAILABLE:
        similarities = cosine_similarity(query_embedding_np, doc_embeddings)[0]
    else:
        norm_query = np.linalg.norm(query_embedding_np)
        norm_docs = np.linalg.norm(doc_embeddings, axis=1)
        dot_products = np.dot(query_embedding_np, doc_embeddings.T)[0]
        similarities = dot_products / (norm_query * norm_docs)
    doc_sim_pairs = list(zip(docs_with_embeddings, similarities))
    sorted_pairs = sorted(doc_sim_pairs, key=lambda x: x[1], reverse=True)
    return sorted_pairs[:top_k]

def create_faiss_index(documents: List[Document]) -> Any:
    if not FAISS_AVAILABLE:
        logger.error("FAISS required for indexing")
        return None
    docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
    if not docs_with_embeddings:
        logger.warning("No documents with embeddings found")
        return None
    embedding_dim = len(docs_with_embeddings[0].embedding)
    index = faiss.IndexFlatL2(embedding_dim)
    embeddings = np.array([doc.embedding for doc in docs_with_embeddings], dtype=np.float32)
    index.add(embeddings)
    return index, docs_with_embeddings

def faiss_similarity_search(
    query_embedding: List[float],
    faiss_index: Any,
    documents: List[Document],
    top_k: int = DEFAULT_TOP_K
) -> List[Tuple[Document, float]]:
    if not FAISS_AVAILABLE:
        logger.error("FAISS required for similarity search")
        return []
    if faiss_index is None:
        logger.error("FAISS index is None")
        return []
    if isinstance(faiss_index, tuple):
        index, docs_with_embeddings = faiss_index
    else:
        index = faiss_index
        docs_with_embeddings = documents
    query_np = np.array([query_embedding], dtype=np.float32)
    distances, indices = index.search(query_np, top_k)
    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx < len(docs_with_embeddings):
            similarity = 1.0 / (1.0 + distances[0][i])
            results.append((docs_with_embeddings[idx], similarity))
    return results

class RAGSystem:
    """
    Base class for Retrieval-Augmented Generation systems.
    Provides common functionality and interfaces.
    """

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        system_message: str = "You are a helpful assistant that answers based on the retrieved context.",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        verbose: bool = False
    ):
        self.client, self.model = setup_client(model=model) if client is None else (client, model)
        self.embedding_model = embedding_model
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.documents = []
        self.history = []
        self.metrics = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "retrieval_latency": 0,
            "queries": 0
        }

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        if metadatas is None:
            metadatas = [{} for _ in texts]
        documents = [
            Document(content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        self.add_documents(documents)

    def _retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError("Subclasses must implement _retrieve")

    def _format_context(
        self,
        retrieved_documents: List[Tuple[Document, float]]
    ) -> str:
        context_parts = []
        for i, (doc, score) in enumerate(retrieved_documents):
            source_info = ""
            if doc.metadata:
                source = doc.metadata.get("source", "")
                if source:
                    source_info = f" (Source: {source})"
            context_parts.append(f"[Document {i+1}{source_info}]\n{doc.content}\n")
        return "\n".join(context_parts)

    def _create_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        return f"""Answer the following question based on the retrieved context. If the context doesn't contain relevant information, say so instead of making up an answer.

Retrieved Context:
{context}

Question: {query}

Answer:"""

    def query(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[str, Dict[str, Any]]:
        self._log(f"Processing query: {query}")
        start_time = time.time()
        retrieved_docs = self._retrieve(query, top_k)
        retrieval_latency = time.time() - start_time
        context = self._format_context(retrieved_docs)
        prompt = self._create_prompt(query, context)
        response, metadata = generate_response(
            prompt=prompt,
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_message=self.system_message
        )
        self.metrics["total_prompt_tokens"] += metadata.get("prompt_tokens", 0)
        self.metrics["total_response_tokens"] += metadata.get("response_tokens", 0)
        self.metrics["total_tokens"] += metadata.get("total_tokens", 0)
        self.metrics["total_latency"] += metadata.get("latency", 0)
        self.metrics["retrieval_latency"] += retrieval_latency
        self.metrics["queries"] += 1
        query_record = {
            "query": query,
            "retrieved_docs": [(doc.content, score) for doc, score in retrieved_docs],
            "context": context,
            "prompt": prompt,
            "response": response,
            "metrics": {
                **metadata,
                "retrieval_latency": retrieval_latency
            },
            "timestamp": time.time()
        }
        self.history.append(query_record)
        details = {
            "query": query,
            "retrieved_docs": [(doc, score) for doc, score in retrieved_docs],
            "context": context,
            "response": response,
            "metrics": {
                **metadata,
                "retrieval_latency": retrieval_latency
            }
        }
        return response, details

    def get_summary_metrics(self) -> Dict[str, Any]:
        summary = self.metrics.copy()
        if summary["queries"] > 0:
            summary["avg_latency_per_query"] = summary["total_latency"] / summary["queries"]
            summary["avg_retrieval_latency"] = summary["retrieval_latency"] / summary["queries"]

        if summary["total_prompt_tokens"] > 0:
            summary["overall_efficiency"] = (
                summary["total_response_tokens"] / summary["total_prompt_tokens"]
            )
        return summary

    def display_query_results(self, details: Dict[str, Any], show_context: bool = True, example_dir: Optional[pathlib.Path] = None) -> None:
        """
        Save the query results to files under a dedicated example directory.
        """
        if example_dir is None:
            example_dir = BASE_OUTPUT_DIR / "example_misc"
        example_dir = pathlib.Path(example_dir)
        display_response(
            prompt=details["query"],
            response=details["response"],
            retrieved_context=details.get("context"),
            metrics=details.get("metrics"),
            show_prompt=True,
            show_context=show_context,
            output_dir=example_dir
        )
        if show_context and "retrieved_docs" in details:
            docs_dir = example_dir / "retrieved_docs"
            for i, (doc, score) in enumerate(details["retrieved_docs"]):
                doc_dir = docs_dir / f"doc_{i+1}_score_{score:.4f}"
                doc_dir.mkdir(parents=True, exist_ok=True)
                (doc_dir / "content.md").write_text(doc.content)
                if doc.metadata:
                    (doc_dir / "metadata.json").write_text(json.dumps(doc.metadata, indent=2))

    def visualize_metrics(self, example_dir: Optional[pathlib.Path] = None) -> None:
        """
        Create visualization of metrics across queries and save to example_dir.
        """
        if not self.history:
            logger.warning("No history to visualize")
            return
        queries = list(range(1, len(self.history) + 1))
        prompt_tokens = [h["metrics"].get("prompt_tokens", 0) for h in self.history]
        response_tokens = [h["metrics"].get("response_tokens", 0) for h in self.history]
        generation_latencies = [h["metrics"].get("latency", 0) for h in self.history]
        retrieval_latencies = [h["metrics"].get("retrieval_latency", 0) for h in self.history]
        total_latencies = [g + r for g, r in zip(generation_latencies, retrieval_latencies)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RAG System Metrics by Query", fontsize=16)

        axes[0, 0].bar(queries, prompt_tokens, label="Prompt Tokens", color="blue", alpha=0.7)
        axes[0, 0].bar(queries, response_tokens, bottom=prompt_tokens, label="Response Tokens", color="green", alpha=0.7)
        axes[0, 0].set_title("Token Usage")
        axes[0, 0].set_xlabel("Query")
        axes[0, 0].set_ylabel("Tokens")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].bar(queries, retrieval_latencies, label="Retrieval", color="orange", alpha=0.7)
        axes[0, 1].bar(queries, generation_latencies, bottom=retrieval_latencies, label="Generation", color="red", alpha=0.7)
        axes[0, 1].set_title("Latency Breakdown")
        axes[0, 1].set_xlabel("Query")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        if any("retrieved_docs" in h for h in self.history):
            doc_counts = [len(h.get("retrieved_docs", [])) for h in self.history]
            axes[1, 0].plot(queries, doc_counts, marker='o', color="purple", alpha=0.7)
            axes[1, 0].set_title("Retrieved Documents Count")
            axes[1, 0].set_xlabel("Query")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].grid(alpha=0.3)

        cumulative_tokens = np.cumsum([h["metrics"].get("total_tokens", 0) for h in self.history])
        axes[1, 1].plot(queries, cumulative_tokens, marker='^', color="brown", alpha=0.7)
        axes[1, 1].set_title("Cumulative Token Usage")
        axes[1, 1].set_xlabel("Query")
        axes[1, 1].set_ylabel("Total Tokens")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if example_dir is None:
            example_dir = BASE_OUTPUT_DIR / "example_misc"
        save_visualization(fig, pathlib.Path(example_dir), name="metrics_visualization")

    def _save_history(self, example_dir: pathlib.Path) -> None:
        (example_dir / "history.json").write_text(json.dumps(self.history, indent=2, default=str))

class SimpleRAG(RAGSystem):
    """
    A simple RAG system that uses embeddings for similarity search.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.documents_embedded = False

    def add_documents(self, documents: List[Document]) -> None:
        super().add_documents(documents)
        self.documents_embedded = False

    def _ensure_documents_embedded(self) -> None:
        if self.documents_embedded:
            return
        docs_to_embed = [doc for doc in self.documents if doc.embedding is None]
        if docs_to_embed:
            self._log(f"Generating embeddings for {len(docs_to_embed)} documents")
            extract_document_batch_embeddings(
                docs_to_embed,
                client=self.client,
                model=self.embedding_model
            )
        self.documents_embedded = True

    def _retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        self._ensure_documents_embedded()
        if not self.documents:
            self._log("No documents in the document store")
            return []
        query_embedding = generate_embedding(
            query,
            client=self.client,
            model=self.embedding_model
        )
        results = similarity_search(
            query_embedding,
            self.documents,
            top_k
        )
        return results

class ChunkedRAG(SimpleRAG):
    """
    A RAG system that chunks documents before indexing.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.original_documents = []
        self.use_faiss = FAISS_AVAILABLE
        self.faiss_index = None

    def add_documents(self, documents: List[Document]) -> None:
        self.original_documents.extend(documents)
        chunked_docs = []
        for doc in documents:
            chunks = text_to_chunks(
                doc.content,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                model=self.model
            )
            for i, chunk in enumerate(chunks):
                chunk.metadata.update(doc.metadata)
                chunk.metadata["parent_id"] = doc.id
                chunk.metadata["chunk_index"] = i
                chunk.metadata["parent_content"] = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            chunked_docs.extend(chunks)
        super().add_documents(chunked_docs)
        if self.use_faiss:
            self.faiss_index = None

    def _ensure_documents_embedded(self) -> None:
        super()._ensure_documents_embedded()
        if self.use_faiss and self.faiss_index is None and self.documents:
            self._log("Building FAISS index")
            self.faiss_index = create_faiss_index(self.documents)

    def _retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        self._ensure_documents_embedded()
        if not self.documents:
            self._log("No documents in the document store")
            return []
        query_embedding = generate_embedding(
            query,
            client=self.client,
            model=self.embedding_model
        )
        if self.use_faiss and self.faiss_index is not None:
            results = faiss_similarity_search(
                query_embedding,
                self.faiss_index,
                self.documents,
                top_k
            )
        else:
            results = similarity_search(
                query_embedding,
                self.documents,
                top_k
            )
        return results

class HybridRAG(ChunkedRAG):
    """
    A RAG system that combines embedding similarity with keyword search.
    """

    def __init__(
        self,
        keyword_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.keyword_weight = max(0.0, min(1.0, keyword_weight))
        self.embedding_weight = 1.0 - self.keyword_weight

    def _keyword_search(
        self,
        query: str,
        documents: List[Document],
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        query_terms = set(query.lower().split())

        results = []
        for doc in documents:
            content = doc.content.lower()
            matches = sum(1 for term in query_terms if term in content)
            score = matches / len(query_terms) if query_terms else 0.0

            results.append((doc, score))
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        self._ensure_documents_embedded()
        if not self.documents:
            self._log("No documents in the document store")
            return []

        query_embedding = generate_embedding(
            query,
            client=self.client,
            model=self.embedding_model
        )

        # 1. Semantic search (embedding-based)
        if self.use_faiss and self.faiss_index is not None:
            semantic_results = faiss_similarity_search(
                query_embedding,
                self.faiss_index,
                self.documents,
                top_k * 2  # Get more candidates for fusion
            )
        else:
            semantic_results = similarity_search(
                query_embedding,
                self.documents,
                top_k * 2
            )

        # 2. Keyword search
        keyword_results = self._keyword_search(query, self.documents, top_k * 2)

        # 3. Normalize scores to [0,1]
        def normalize(scores: List[float]) -> List[float]:
            if not scores:
                return []
            min_s, max_s = min(scores), max(scores)
            return [(s - min_s) / (max_s - min_s) if max_s > min_s else 1.0 for s in scores]

        semantic_docs, semantic_scores = zip(*semantic_results) if semantic_results else ([], [])
        keyword_docs, keyword_scores = zip(*keyword_results) if keyword_results else ([], [])

        semantic_scores_norm = normalize(semantic_scores)
        keyword_scores_norm = normalize(keyword_scores)

        # 4. Combine using weighted sum
        combined: Dict[Document, float] = {}
        for doc, score in zip(semantic_docs, semantic_scores_norm):
            combined[doc] = combined.get(doc, 0.0) + self.embedding_weight * score
        for doc, score in zip(keyword_docs, keyword_scores_norm):
            combined[doc] = combined.get(doc, 0.0) + self.keyword_weight * score

        # 5. Sort and return top_k
        final_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return final_results

# -----------------------------------------------------------------------------
# Example Demonstrations (run when module is executed directly)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import textwrap

    # Sample documents
    sample_texts = [
        textwrap.dedent("""\
            Python is a high-level, interpreted programming language known for its simplicity
            and readability. Created by Guido van Rossum and first released in 1991, Python's
            design philosophy emphasizes code readability with its notable use of significant
            whitespace.
            """),
        textwrap.dedent("""\
            Retrieval-Augmented Generation (RAG) combines information retrieval with text
            generation. It retrieves relevant documents from a knowledge base and feeds them
            into a language model to produce more accurate and contextually grounded responses.
            """),
        textwrap.dedent("""\
            FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
            and clustering of dense vectors. It contains algorithms that search in sets of
            vectors of any size, up to ones that possibly do not fit in RAM.
            """)
    ]

    # Example 1: SimpleRAG
    example_1_dir = BASE_OUTPUT_DIR / "example_1_simple_rag"
    rag1 = SimpleRAG(verbose=True)
    rag1.add_texts(sample_texts)
    resp1, det1 = rag1.query("What is Python known for?")
    rag1.display_query_results(det1, show_context=True, example_dir=example_1_dir)
    rag1.visualize_metrics(example_dir=example_1_dir)
    rag1._save_history(example_1_dir)

    # Example 2: ChunkedRAG with larger document
    example_2_dir = BASE_OUTPUT_DIR / "example_2_chunked_rag"
    long_doc = "\n\n".join([
        "Section 1: Machine learning is a subset of artificial intelligence...",
        "Section 2: Deep learning utilizes neural networks with many layers...",
        "Section 3: Transformers revolutionized NLP in 2017...",
        "Section 4: Embedding models convert text into dense vectors..."
    ])
    rag2 = ChunkedRAG(chunk_size=200, chunk_overlap=50, verbose=True)
    rag2.add_texts([long_doc], metadatas=[{"source": "ml_guide.txt"}])
    resp2, det2 = rag2.query("How do transformers work in NLP?")
    rag2.display_query_results(det2, show_context=True, example_dir=example_2_dir)
    rag2.visualize_metrics(example_dir=example_2_dir)
    rag2._save_history(example_2_dir)

    # Example 3: HybridRAG
    example_3_dir = BASE_OUTPUT_DIR / "example_3_hybrid_rag"
    rag3 = HybridRAG(keyword_weight=0.4, verbose=True)
    rag3.add_texts(sample_texts + [long_doc])
    resp3, det3 = rag3.query("FAISS library")
    rag3.display_query_results(det3, show_context=True, example_dir=example_3_dir)
    rag3.visualize_metrics(example_dir=example_3_dir)
    rag3._save_history(example_3_dir)

    # Summary
    summary_path = BASE_OUTPUT_DIR / "SUMMARY.md"
    summary_path.write_text(textwrap.dedent(f"""\
        # RAG Recipes Demonstrations

        All examples saved under `{BASE_OUTPUT_DIR.resolve()}`

        - **example_1_simple_rag/** – Basic embedding-based retrieval
        - **example_2_chunked_rag/** – Document chunking + optional FAISS
        - **example_3_hybrid_rag/** – Embedding + keyword hybrid search

        Each folder contains:
        - `query.md`, `context.md`, `response.md`
        - `metrics.txt`
        - `retrieved_docs/` with individual documents
        - `metrics_visualization.png/svg`
        - `history.json`
        """))

    print(f"All example outputs saved to: {BASE_OUTPUT_DIR.resolve()}")
