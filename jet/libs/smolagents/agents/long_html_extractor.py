import json
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    tool,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
# shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 8000,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


# ===================
#   Custom Tools
# ===================


@tool
def extract_relevant_passages(
    chunk_text: str,
    query: str,
    min_relevance_score: float = 0.6,
) -> list[dict[str, Any]]:
    """Extract relevant passages from a text chunk that match the given query.

    This is currently a simple keyword-based heuristic. In production, replace
    with LLM-based extraction or semantic search.

    Args:
        chunk_text: The text content of the current HTML/text chunk to analyze
        query: The user query or topic to find relevant content for
        min_relevance_score: Minimum dummy relevance score required to include a passage (default: 0.6)

    Returns:
        List of dictionaries containing extracted passages, each with:
            - text (str): the matching passage/sentence
            - relevance_score (float): computed relevance (0–1)
            - start_idx (int): character start position within the chunk
            - end_idx (int): character end position within the chunk
    """
    # This is a placeholder — in real usage, a dedicated extraction agent would be called here
    # For simplicity we simulate extraction logic
    # In production → replace with a real CodeAgent or ToolCallingAgent call

    passages = []
    sentences = chunk_text.split(". ")
    for i, sent in enumerate(sentences):
        # Dummy relevance: higher if query words appear
        score = sum(word.lower() in sent.lower() for word in query.split()) / max(
            1, len(query.split())
        )
        score = min(score * 1.2, 0.99)  # cap at ~1.0

        if score >= min_relevance_score:
            passages.append(
                {
                    "text": sent.strip() + ".",
                    "relevance_score": round(score, 3),
                    "start_idx": chunk_text.find(sent),
                    "end_idx": chunk_text.find(sent) + len(sent) + 1,
                }
            )

    return passages


@tool
def chunk_html(
    html_content: str,
    chunk_size: int = 4000,
    overlap: int = 800,
) -> list[dict[str, str]]:
    """Split cleaned HTML text into overlapping chunks suitable for LLM processing.

    Args:
        html_content: Raw HTML string to process
        chunk_size: Approximate maximum characters per chunk (default: 4000)
        overlap: Number of characters to overlap between consecutive chunks (default: 800)

    Returns:
        List of dictionaries, each representing one chunk:
            - chunk_id (int): sequential index starting from 0
            - text (str): cleaned text content of this chunk
            - start_char (int): starting character position in the full cleaned text
            - end_char (int): ending character position in the full cleaned text
    """
    soup = BeautifulSoup(html_content, "lxml")
    text = soup.get_text(separator=" ", strip=True)

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        chunks.append(
            {
                "chunk_id": len(chunks),
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
            }
        )

        start += chunk_size - overlap

    return chunks


@tool
def save_partial_results(
    results: list[dict[str, Any]],
    filename: str = "partial_results.json",
) -> str:
    """Save the current list of extracted passages/results to a JSON file.

    Args:
        results: List of passage dictionaries to save
        filename: Path to the output JSON file (default: "partial_results.json")

    Returns:
        Status message indicating success or error
    """
    try:
        with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return f"Saved {len(results)} partial results to {filename}"
    except Exception as e:
        return f"Error saving: {str(e)}"


# ===================
#   Progress Tracker (simple class - can be agentized later)
# ===================


class ProgressTracker:
    def __init__(self, total_chunks: int):
        self.total_chunks = total_chunks
        self.processed = 0
        self.results: list[dict[str, Any]] = []
        self.checkpoint_file = "extraction_checkpoint.json"

    def update(self, chunk_id: int, new_results: list[dict[str, Any]]):
        self.processed += 1
        self.results.extend(new_results)

        # Optional: save every 3 chunks
        if self.processed % 3 == 0 or self.processed == self.total_chunks:
            self.save_checkpoint()

        progress_pct = (self.processed / self.total_chunks) * 100
        print(
            f"\n[PROGRESS] Chunk {chunk_id + 1}/{self.total_chunks} | {progress_pct:.1f}% | "
            f"Collected {len(self.results)} passages so far"
        )

    def save_checkpoint(self):
        data = {
            "processed_chunks": self.processed,
            "total_chunks": self.total_chunks,
            "partial_results": self.results,
        }
        with open(OUTPUT_DIR / self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Checkpoint saved: {len(self.results)} passages")

    def get_final_results(self) -> list[dict[str, Any]]:
        # Simple deduplication by text (can be improved with embeddings)
        seen = set()
        unique = []
        for r in self.results:
            if r["text"] not in seen:
                seen.add(r["text"])
                unique.append(r)
        return sorted(unique, key=lambda x: x["relevance_score"], reverse=True)


# ===================
#   Orchestrator Agent
# ===================


def run_long_html_extraction(
    html_url_or_path: str,
    query: str,
    model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
    chunk_size: int = 4000,
    overlap: int = 800,
    use_hf_inference: bool = False,
) -> list[dict[str, Any]]:
    """Main function: Extract all relevant documents/passages from long HTML using sliding window + agents."""

    # Load HTML
    if html_url_or_path.startswith(("http://", "https://")):
        response = requests.get(html_url_or_path, timeout=15)
        response.raise_for_status()
        html_content = response.text
    else:
        with open(html_url_or_path, encoding="utf-8") as f:
            html_content = f.read()

    # Prepare model
    if use_hf_inference:
        model = InferenceClientModel(model_id=model_id)
    else:
        # Example using LiteLLM (OpenAI-compatible endpoint)
        model = create_local_model(agent_name="worker_agent")

    # Create worker agent (can be specialized further)
    worker_agent = CodeAgent(
        tools=[extract_relevant_passages],
        model=model,
        max_steps=5,
        add_base_tools=False,
    )

    # Create chunking tool instance
    chunk_tool = chunk_html

    # Chunk the document
    chunks = chunk_tool(
        html_content=html_content, chunk_size=chunk_size, overlap=overlap
    )
    print(f"Split document into {len(chunks)} overlapping chunks")

    # Progress & result collector
    tracker = ProgressTracker(total_chunks=len(chunks))

    # Main loop: process each chunk
    for chunk in chunks:
        print(
            f"\nProcessing chunk {chunk['chunk_id'] + 1} ({chunk['start_char']}-{chunk['end_char']})..."
        )

        # Ask worker agent to extract relevant parts
        task = (
            f"Extract **all** passages from the following text chunk that are relevant to the query: '{query}'\n\n"
            f"Text chunk:\n{chunk['text']}\n\n"
            "Return ONLY the list of extracted passages using the extract_relevant_passages tool."
        )

        try:
            result = worker_agent.run(task)

            if isinstance(result, list):
                # Assume this is already the list of passages returned by the tool
                passages = result
                if passages and all(isinstance(p, dict) for p in passages):
                    # Add metadata from the current chunk
                    for p in passages:
                        p["chunk_id"] = chunk["chunk_id"]
                        p["source_start"] = chunk["start_char"] + p.get("start_idx", 0)
                        p["source_end"] = chunk["start_char"] + p.get("end_idx", 0)
                    tracker.update(chunk["chunk_id"], passages)
                else:
                    print(
                        f"Warning: Agent returned list but not list[dict] → {result[:1]}"
                    )
            elif result is None:
                print("Agent returned None — no extraction performed")
            else:
                print(f"Unexpected result from agent.run(): {type(result)} → {result}")
        except Exception as e:
            print(f"Error processing chunk {chunk['chunk_id']}: {e}")

    # Optional final save
    save_partial_results(tracker.results, "final_results.json")

    return tracker.get_final_results()


# ===================
#   Example Usage
# ===================

if __name__ == "__main__":
    # Example: extract information about supported tasks / pipelines from HF Transformers docs
    url = "https://huggingface.co/docs/transformers/index"
    # Alternative good long pages you can try:
    # url = "https://huggingface.co/docs/transformers/en/main_classes/pipelines"
    # url = "https://github.com/huggingface/transformers/blob/main/README.md"
    # url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    query = "pipeline OR task OR supported task OR text-classification OR token-classification OR question-answering OR summarization OR translation OR text-generation"

    print("Starting long HTML extraction...")
    results = run_long_html_extraction(
        html_url_or_path=url,
        query=query,
        chunk_size=3500,
        overlap=700,
        use_hf_inference=False,  # ← change to False if using local model (Ollama/vLLM/llama.cpp)
    )

    print(f"\nFound {len(results)} relevant passages:")
    print("-" * 80)

    # Show top 10 most relevant passages (or all if fewer)
    for i, r in enumerate(results[:10], 1):
        print(
            f"{i}. Relevance: {r['relevance_score']:.2f}  |  Chunk {r.get('chunk_id', '?')}"
        )
        print(f"    {r['text'][:280]}{'...' if len(r['text']) > 280 else ''}")
        print()
