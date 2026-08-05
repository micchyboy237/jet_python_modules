# /Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/adapters/llama_cpp/examples/chunk_strategies/01_demo_sentence_chunking.py
"""Demo: TokenAwareSentenceChunker for prose with mixed-length sentences.

This input contains short definitions, a very long explanatory sentence that
exceeds typical chunk budgets, and list-like structures. The sentence strategy
preserves semantic boundaries, splits the oversized sentence at word level
without breaking mid-phrase, and applies token-exact overlap so context
continuity is maintained across chunk boundaries.
"""

from jet.adapters.llama_cpp.chunk_strategy_utils import get_chunker

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


def main() -> None:
    chunker = get_chunker("sentence", model="qwen3.5:2b")

    chunks = chunker.chunk(
        text=TEXT,
        chunk_size=64,
        chunk_overlap=12,
        min_chunk_size=16,
        buffer=4,
    )

    print(f"Strategy: TokenAwareSentenceChunker")
    print(f"Input length: {len(TEXT)} chars")
    print(f"Output chunks: {len(chunks)}")
    print("-" * 60)

    for i, chunk in enumerate(chunks):
        token_count = len(chunker.size_fn(chunk))
        print(f"\n[Chunk {i}] ({token_count} tokens)")
        print(chunk)

    # Verify no chunk exceeds effective budget
    effective = 64 - 4
    oversized = [
        (i, len(chunker.size_fn(c)))
        for i, c in enumerate(chunks)
        if len(chunker.size_fn(c)) > effective
    ]
    print("\n" + "-" * 60)
    if oversized:
        print(f"WARNING: {len(oversized)} chunk(s) exceed effective size {effective}:")
        for idx, count in oversized:
            print(f"  Chunk {idx}: {count} tokens")
    else:
        print(f"All chunks within effective budget of {effective} tokens.")


if __name__ == "__main__":
    main()
