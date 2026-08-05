# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/01_demo_sentence_chunking.py
"""Demo: TokenAwareSentenceChunker with overlap visualization.

Demonstrates sentence-aware chunking with token-exact overlap on prose
containing short definitions, a very long explanatory sentence, and
list-like structures. Overlap regions between consecutive chunks are
highlighted to verify semantic continuity at boundaries.
"""

from jet.adapters.llama_cpp.chunk_strategies import detect_text_overlap, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL

TEXT = (
    "Retrieval-augmented generation combines external knowledge with language models. "
    "It reduces hallucination by grounding responses in verified source material. "
    "The architecture typically consists of three components: a document ingestion pipeline "
    "that chunks and embeds text into a vector store, a retrieval module that selects "
    "the most relevant chunks based on semantic similarity to the user query, and a "
    "generation module that conditions the language model on both the query and the "
    "retrieved context to produce a faithful, cited answer that stays within the "
    "model's context window while maximizing information density per token. "
    "Chunking strategy directly impacts retrieval precision. "
    "Poorly chosen boundaries introduce noise that degrades answer quality."
)

CHUNK_SIZE = 64
CHUNK_OVERLAP = 12
MIN_CHUNK_SIZE = 16
BUFFER = 4


def main() -> None:
    chunker = get_chunker("sentence", model=LLM_MODEL)

    chunks = chunker.chunk(
        text=TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
    )

    effective = CHUNK_SIZE - BUFFER

    print(f"Strategy: TokenAwareSentenceChunker")
    print(
        f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
        f"min={MIN_CHUNK_SIZE}, buffer={BUFFER} → effective={effective}"
    )
    print(f"Input length: {len(TEXT)} chars")
    print(f"Output chunks: {len(chunks)}")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        token_count = len(chunker.size_fn(chunk))
        print(f"\n[Chunk {i}] ({token_count} tokens)")
        print(chunk)

        # Show overlap with next chunk
        if i < len(chunks) - 1:
            overlap_text, overlap_tokens = detect_text_overlap(
                chunk, chunks[i + 1], chunker.size_fn
            )
            if overlap_text and overlap_tokens > 0:
                print(f"  ↕ Overlap with Chunk {i + 1}: {overlap_tokens} tokens")
                print(f'    "{overlap_text}"')
            else:
                print(f"  ↕ No overlap detected with Chunk {i + 1}")

    # Budget validation
    print("\n" + "=" * 60)
    oversized = [
        (i, len(chunker.size_fn(c)))
        for i, c in enumerate(chunks)
        if len(chunker.size_fn(c)) > effective
    ]
    if oversized:
        print(
            f"⚠️  WARNING: {len(oversized)} chunk(s) exceed effective size {effective}:"
        )
        for idx, count in oversized:
            print(f"   Chunk {idx}: {count} tokens")
    else:
        print(f"✅ All chunks within effective budget of {effective} tokens.")


if __name__ == "__main__":
    main()
