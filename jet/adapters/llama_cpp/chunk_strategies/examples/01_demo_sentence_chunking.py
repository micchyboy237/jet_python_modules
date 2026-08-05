"""Demo: TokenAwareSentenceChunker with overlap visualization.
Demonstrates sentence-aware chunking with token-exact overlap on prose
containing short definitions, a very long explanatory sentence, and
list-like structures. Overlap regions between consecutive chunks are
highlighted to verify semantic continuity at boundaries.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.chunk_strategies import detect_text_overlap, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL

# Rich console for styled resource links
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Build results lines for saving
    results_lines = [
        f"Strategy: TokenAwareSentenceChunker",
        f"Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
        f"min={MIN_CHUNK_SIZE}, buffer={BUFFER} → effective={effective}",
        f"Input length: {len(TEXT)} chars",
        f"Output chunks: {len(chunks)}",
        "=" * 60,
    ]

    for i, chunk in enumerate(chunks):
        token_count = len(chunker.size_fn(chunk))
        chunk_header = f"\n[Chunk {i}] ({token_count} tokens)"
        print(chunk_header)
        print(chunk)
        results_lines.append(chunk_header)
        results_lines.append(chunk)

        if i < len(chunks) - 1:
            overlap_text, overlap_tokens = detect_text_overlap(
                chunk, chunks[i + 1], chunker.size_fn
            )
            if overlap_text and overlap_tokens > 0:
                overlap_msg = f"  ↕ Overlap with Chunk {i + 1}: {overlap_tokens} tokens"
                overlap_detail = f'    "{overlap_text}"'
                print(overlap_msg)
                print(overlap_detail)
                results_lines.append(overlap_msg)
                results_lines.append(overlap_detail)
            else:
                no_overlap_msg = f"  ↕ No overlap detected with Chunk {i + 1}"
                print(no_overlap_msg)
                results_lines.append(no_overlap_msg)

    print("\n" + "=" * 60)
    results_lines.append("\n" + "=" * 60)

    oversized = [
        (i, len(chunker.size_fn(c)))
        for i, c in enumerate(chunks)
        if len(chunker.size_fn(c)) > effective
    ]
    if oversized:
        warning_msg = (
            f"⚠️  WARNING: {len(oversized)} chunk(s) exceed effective size {effective}:"
        )
        print(warning_msg)
        results_lines.append(warning_msg)
        for idx, count in oversized:
            detail_msg = f"   Chunk {idx}: {count} tokens"
            print(detail_msg)
            results_lines.append(detail_msg)
    else:
        success_msg = f"✅ All chunks within effective budget of {effective} tokens."
        print(success_msg)
        results_lines.append(success_msg)

    # Save results
    summary_path = OUTPUT_DIR / "chunking_results.txt"
    summary_path.write_text("\n".join(results_lines), encoding="utf-8")
    console.print(
        f"💾 Results saved to [bold blue][link=file://{summary_path}]{summary_path.name}[/link][/bold blue]"
    )

    # Save individual chunks
    chunks_dir = OUTPUT_DIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        chunk_path = chunks_dir / f"chunk_{i:02d}.txt"
        chunk_path.write_text(chunk, encoding="utf-8")
    console.print(
        f"💾 Individual chunks saved to [bold blue][link=file://{chunks_dir}]{chunks_dir.name}/[/link][/bold blue] "
        f"({len(chunks)} files)"
    )


if __name__ == "__main__":
    main()
