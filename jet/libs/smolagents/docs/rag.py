# demo_agentic_rag_local_no_langchain.py
"""
Demonstration of Agentic RAG using smolagents with LOCAL llama.cpp server
No LangChain dependencies
"""

import re
import time
from typing import Any, Dict, List

import datasets
from jet.libs.smolagents.utils.model_utils import create_local_model
from rank_bm25 import BM25Okapi
from smolagents import CodeAgent, Tool

# ──────────────────────────────────────────────────────────────────────────────
# Simple text splitter (no LangChain)
# ──────────────────────────────────────────────────────────────────────────────


def simple_recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None,
) -> List[str]:
    if separators is None:
        separators = ["\n\n", "\n", ".", " ", ""]

    def _split(text: str, level: int = 0) -> List[str]:
        if level >= len(separators):
            # Final fallback: split by fixed size
            return [
                text[i : i + chunk_size]
                for i in range(0, len(text), chunk_size - chunk_overlap)
            ]

        sep = separators[level]
        if not sep:
            return _split(text, level + 1)

        parts = re.split(re.escape(sep), text)
        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= chunk_size:
                current += (sep if current else "") + part
            else:
                if current:
                    chunks.append(current)
                current = part
                # handle overlap
                if len(current) > chunk_size:
                    # too big even alone → force split
                    while len(current) > chunk_size:
                        chunks.append(current[:chunk_size])
                        current = current[chunk_size - chunk_overlap :]

        if current:
            chunks.append(current)

        # merge very small trailing chunks
        final = []
        buffer = ""
        for chunk in chunks:
            if len(buffer) + len(chunk) <= chunk_size + chunk_overlap:
                buffer += chunk
            else:
                if buffer:
                    final.append(buffer)
                buffer = chunk
        if buffer:
            final.append(buffer)

        return final

    return _split(text)


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge base & tool preparation
# ──────────────────────────────────────────────────────────────────────────────


def prepare_knowledge_base_and_tool():
    """Load, filter, split HF docs and create BM25 retriever tool (run once)."""
    print("Loading & preparing knowledge base... (one-time)")

    # Load dataset
    ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    # Keep only transformers docs
    ds = ds.filter(lambda row: row["source"].startswith("huggingface/transformers"))

    # Simple documents
    source_docs: List[Dict[str, Any]] = [
        {"content": doc["text"], "metadata": {"source": doc["source"].split("/")[1]}}
        for doc in ds
    ]

    print(f"→ Loaded {len(source_docs)} source documents")

    # Split
    docs_processed = []
    for doc in source_docs:
        chunks = simple_recursive_split(
            doc["content"],
            chunk_size=500,
            chunk_overlap=50,
        )
        for i, chunk in enumerate(chunks):
            docs_processed.append(
                {
                    "content": chunk.strip(),
                    "metadata": {**doc["metadata"], "chunk_id": i},
                }
            )

    print(f"→ Prepared {len(docs_processed)} document chunks")

    # BM25
    tokenized_corpus = [chunk["content"].lower().split() for chunk in docs_processed]
    bm25 = BM25Okapi(tokenized_corpus)

    class RetrieverTool(Tool):
        name = "retriever"
        description = (
            "Uses lexical (BM25) search to retrieve relevant parts of Hugging Face "
            "Transformers documentation. Input should be affirmative statements, "
            "not questions. Use natural phrasing close to documentation style."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "Search query – preferably keyword-rich and affirmative.",
            }
        }
        output_type = "string"

        def __init__(self, documents: List[Dict], bm25_engine, **kwargs):
            super().__init__(**kwargs)
            self.documents = documents
            self.bm25 = bm25_engine

        def forward(self, query: str) -> str:
            if not query or not query.strip():
                return "No query provided."

            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_top_n(tokenized_query, self.documents, n=6)

            if not doc_scores:
                return "No relevant documents found."

            formatted = "\nRetrieved documents:\n"
            for i, doc in enumerate(doc_scores):
                src = doc["metadata"].get("source", "unknown")
                formatted += (
                    f"\n\n===== Document {i + 1}  [{src}] =====\n{doc['content']}\n"
                )

            return formatted

    tool = RetrieverTool(docs_processed, bm25)
    print("→ RetrieverTool ready")
    return tool


# Prepare once (global)
RETRIEVER_TOOL = prepare_knowledge_base_and_tool()


def create_rag_agent(
    max_steps: int = 5,
    verbosity_level: int = 2,
    temperature: float = 0.6,
) -> CodeAgent:
    model = create_local_model(temperature=temperature)
    return CodeAgent(
        tools=[RETRIEVER_TOOL],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demos (unchanged)
# ──────────────────────────────────────────────────────────────────────────────


def demo_rag_1_simple_question():
    print("\n" + "=" * 70)
    print("Demo 1: Simple question → forward vs backward pass")
    print("=" * 70)

    agent = create_rag_agent(max_steps=4, verbosity_level=2)

    question = (
        "For a transformers model training, which is slower, "
        "the forward or the backward pass?"
    )

    print(f"\nQuestion: {question}\n")
    start = time.time()
    answer = agent.run(question)
    print(f"\nFinal answer (took {time.time() - start:.1f}s):\n{answer}")


def demo_rag_2_multi_step_reasoning():
    print("\n" + "=" * 70)
    print("Demo 2: Multi-step reasoning question")
    print("=" * 70)

    agent = create_rag_agent(max_steps=7, verbosity_level=2, temperature=0.65)

    question = (
        "How does the Trainer class in transformers handle gradient accumulation? "
        "Show the relevant parameters and explain when to use it."
    )

    print(f"\nQuestion: {question}\n")
    start = time.time()
    answer = agent.run(question)
    print(f"\nFinal answer (took {time.time() - start:.1f}s):\n{answer}")


def main():
    print("=" * 78)
    print("  Agentic RAG Demos  —  LOCAL llama.cpp server  ".center(78))
    print("  No LangChain • BM25 retriever on HF Transformers docs  ".center(78))
    print("=" * 78 + "\n")

    demo_rag_1_simple_question()
    # demo_rag_2_multi_step_reasoning()

    print("\n" + "=" * 78)
    print("Done".center(78))
    print("=" * 78)


if __name__ == "__main__":
    main()
