# demo_agentic_rag_local.py
"""
Demonstration of Agentic RAG using smolagents with LOCAL llama.cpp server
Reuses create_local_model() from previous examples
"""

import time

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from smolagents import CodeAgent, Tool, OpenAIModel

# Reuse from previous file (you can import it if separated)
def create_local_model(
    temperature: float = 0.7,
    max_tokens: int | None = None,
    model_id: str = "local-model",
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        base_url="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Knowledge base & tool preparation (run once)
# ──────────────────────────────────────────────────────────────────────────────

def prepare_knowledge_base_and_tool():
    """Load, filter, split HF docs and create BM25 retriever tool (run once)."""
    print("Loading & preparing knowledge base... (one-time)")

    # Load dataset
    knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

    # Keep only transformers docs
    knowledge_base = knowledge_base.filter(
        lambda row: row["source"].startswith("huggingface/transformers")
    )

    # To langchain Documents
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
        for doc in knowledge_base
    ]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    docs_processed = text_splitter.split_documents(source_docs)

    print(f"→ Prepared {len(docs_processed)} document chunks")

    # Create retriever tool
    class RetrieverTool(Tool):
        name = "retriever"
        description = (
            "Uses lexical (BM25) search to retrieve relevant parts of Hugging Face "
            "Transformers documentation. Input should be affirmative statements, "
            "not questions."
        )
        inputs = {
            "query": {
                "type": "string",
                "description": "Search query – make it close to document phrasing.",
            }
        }
        output_type = "string"

        def __init__(self, docs, **kwargs):
            super().__init__(**kwargs)
            self.retriever = BM25Retriever.from_documents(docs, k=6)

        def forward(self, query: str) -> str:
            assert isinstance(query, str)
            docs = self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            formatted = "\nRetrieved documents:\n" + "".join(
                f"\n\n===== Document {i+1} =====\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
            return formatted

    tool = RetrieverTool(docs_processed)
    print("→ RetrieverTool ready")
    return tool


# Prepare once (global)
RETRIEVER_TOOL = prepare_knowledge_base_and_tool()


def create_rag_agent(
    max_steps: int = 5,
    verbosity_level: int = 2,
    temperature: float = 0.6,
) -> CodeAgent:
    """Factory for creating agent with local model + retriever tool."""
    model = create_local_model(temperature=temperature)
    return CodeAgent(
        tools=[RETRIEVER_TOOL],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────

def demo_rag_1_simple_question():
    """Demo 1: Basic Agentic RAG question"""
    print("\n" + "="*70)
    print("Demo 1: Simple question → forward vs backward pass")
    print("="*70)

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
    """Demo 2: Question that benefits from multi-step retrieval & reasoning"""
    print("\n" + "="*70)
    print("Demo 2: Multi-step reasoning question")
    print("="*70)

    agent = create_rag_agent(max_steps=7, verbosity_level=2, temperature=0.65)

    question = (
        "How does the Trainer class in transformers handle gradient accumulation? "
        "Show the relevant parameters and explain when to use it."
    )

    print(f"\nQuestion: {question}\n")
    start = time.time()
    answer = agent.run(question)
    print(f"\nFinal answer (took {time.time() - start:.1f}s):\n{answer}")


def demo_rag_3_show_retrieved_docs():
    """Demo 3: Run agent and show what documents were retrieved in last step"""
    print("\n" + "="*70)
    print("Demo 3: Inspect retrieved documents from last step")
    print("="*70)

    agent = create_rag_agent(max_steps=5, verbosity_level=1)  # lower verbosity

    question = "What is Deepspeed integration in transformers and how do I enable it?"

    print(f"\nQuestion: {question}\n")
    answer = agent.run(question)

    # Try to show last retrieval (heuristic: look in memory for last tool call output)
    if agent.memory.steps:
        last_step = agent.memory.steps[-1]
        if hasattr(last_step, "observations") and isinstance(last_step.observations, str):
            if "Retrieved documents" in last_step.observations:
                print("\nLast retrieved content (from final step):\n")
                print(last_step.observations)
            else:
                print("\nNo clear retrieval output in last step observations.")
        else:
            print("\nLast step has no observations string to show.")

    print(f"\nFinal answer:\n{answer}")


def main():
    print("=" * 78)
    print("  Agentic RAG Demos  —  LOCAL llama.cpp server  ".center(78))
    print("  Using BM25 retriever on HuggingFace Transformers docs  ".center(78))
    print("=" * 78 + "\n")

    # Uncomment what you want to run
    demo_rag_1_simple_question()
    # demo_rag_2_multi_step_reasoning()
    # demo_rag_3_show_retrieved_docs()

    print("\n" + "="*78)
    print("Done".center(78))
    print("="*78)


if __name__ == "__main__":
    main()