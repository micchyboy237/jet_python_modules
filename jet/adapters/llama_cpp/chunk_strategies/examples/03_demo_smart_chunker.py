# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/03_demo_smart_chunker.py
"""Demo: SmartChunker adaptive strategy selection.

Demonstrates structure detection and automatic delegation to sentence or
fixed-size chunkers based on document content. Tests three document types:
narrative prose, Python code, and structured markdown with headers.
"""

import logging

from jet.adapters.llama_cpp.chunk_strategies import estimate_tokens_safe, get_chunker

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

NARRATIVE_TEXT = (
    "Retrieval-augmented generation combines external knowledge with language models. "
    "It reduces hallucination by grounding responses in verified source material. "
    "The architecture typically consists of three components: a document ingestion pipeline "
    "that chunks and embeds text into a vector store, a retrieval module that selects "
    "the most relevant chunks based on semantic similarity to the user query, and a "
    "generation module that conditions the language model on both the query and the "
    "retrieved context to produce a faithful, cited answer. "
    "Chunking strategy directly impacts retrieval precision. "
    "Poorly chosen boundaries introduce noise that degrades answer quality."
)

CODE_TEXT = """\
def compute_attention_scores(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
"""

STRUCTURED_TEXT = """\
# Installation Guide

## Prerequisites

You need Python 3.10+ and CUDA 11.8 installed. Verify your setup before proceeding.

## Step 1: Clone the Repository

Use git to clone the project and install dependencies via pip.

## Step 2: Configure Environment Variables

Set the following variables in your .env file for database and API access.

## Step 3: Run Migrations

Execute the migration script to initialize the database schema.

## Step 4: Start the Server

Launch the application with the production configuration flag enabled.

## Troubleshooting

If you encounter CUDA out-of-memory errors, reduce the batch size in config.yaml.
Check the logs directory for detailed error traces and stack dumps.
"""

CHUNK_SIZE = 64
CHUNK_OVERLAP = 12
MIN_CHUNK_SIZE = 16
BUFFER = 4


def _demo_document(label: str, text: str) -> None:
    """Run SmartChunker on a single document and report results."""
    print(f"\n{'=' * 60}")
    print(f"📄 {label}")
    print(
        f"   Input: {len(text)} chars, ~{estimate_tokens_safe(text, 'qwen3.5:2b')} tokens"
    )
    print(f"{'=' * 60}")

    chunker = get_chunker("smart", model="qwen3.5:2b")
    chunks = chunker.chunk(
        text=text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
    )

    print(f"   Output: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        tok_count = estimate_tokens_safe(chunk, "qwen3.5:2b")
        preview = chunk[:80].replace("\n", "\\n")
        print(f"   [{i}] ({tok_count} tok) {preview}...")


def main() -> None:
    logger.info("=== SmartChunker Adaptive Strategy Demo ===")
    _demo_document("Narrative Prose", NARRATIVE_TEXT)
    _demo_document("Python Code", CODE_TEXT)
    _demo_document("Structured Markdown", STRUCTURED_TEXT)
    print(f"\n{'=' * 60}")
    logger.info(
        "Demo complete. Review DEBUG logs above for structure detection decisions."
    )


if __name__ == "__main__":
    main()
