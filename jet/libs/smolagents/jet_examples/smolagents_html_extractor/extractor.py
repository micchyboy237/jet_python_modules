from smolagents import CodeAgent, Tool
from typing import List, Dict, Any
import requests
from .tools import chunk_html, extract_relevant_content, format_final_results
from .checkpoint import CheckpointManager

def run_html_extraction_pipeline(
    url_or_html: str,
    query: str,
    window_size: int = 4000,
    overlap: int = 800,
    resume: bool = True
) -> str:
    """
    Main function to extract relevant content from long HTML with progress tracking
    """
    checkpoint = CheckpointManager()

    # Load or fetch HTML
    if url_or_html.startswith(("http://", "https://")):
        print(f"Fetching URL: {url_or_html}")
        response = requests.get(url_or_html, timeout=15)
        response.raise_for_status()
        html_content = response.text
    else:
        html_content = url_or_html

    # Chunking
    print("Chunking HTML...")
    chunks = chunk_html(html_content, window_size, overlap)
    print(f"Created {len(chunks)} chunks")

    # Load previous state if resuming
    start_idx = 0
    all_results = []

    if resume:
        progress = checkpoint.load_progress()
        if progress:
            start_idx = progress["processed_chunks"]
            all_results = checkpoint.load_results()
            print(f"Resuming from chunk {start_idx}/{len(chunks)}")

    # Process chunks
    for i in range(start_idx, len(chunks)):
        chunk = chunks[i]
        print(f"Processing chunk {i+1}/{len(chunks)} (characters {chunk['start_char']}-{chunk['end_char']})")

        partial = extract_relevant_content(chunk["text"], query)
        for item in partial:
            item["chunk_index"] = i

        all_results.extend(partial)

        # Save progress after each chunk
        checkpoint.save_partial_results(all_results)
        checkpoint.save_progress(i + 1, len(chunks))

        print(f"→ Found {len(partial)} items in chunk {i+1}")

    # Final result
    final_output = format_final_results(all_results)
    print("\nExtraction complete!")
    return final_output


# --- For smolagents integration (optional) ---

chunk_tool = Tool(
    name="chunk_html",
    description="Split HTML into overlapping text chunks",
    func=chunk_html,
    input_schema={
        "html_content": {"type": "string"},
        "window_size": {"type": "integer", "default": 4000},
        "overlap": {"type": "integer", "default": 800}
    }
)

extract_tool = Tool(
    name="extract_relevant",
    description="Extract relevant sentences from a text chunk based on query",
    func=extract_relevant_content,
    input_schema={
        "chunk_text": {"type": "string"},
        "query": {"type": "string"},
        "max_items_per_chunk": {"type": "integer", "default": 5}
    }
)

# You could then create a CodeAgent with these tools if you want
# agent = CodeAgent(tools=[chunk_tool, extract_tool], model=...)
