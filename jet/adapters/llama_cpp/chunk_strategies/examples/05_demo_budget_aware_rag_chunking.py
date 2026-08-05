"""Demo: Budget-aware RAG chunking pipeline.
Demonstrates end-to-end safe RAG: chunking → retrieval-type overlap →
budget validation → guaranteed-safe prompt assembly. Shows how PromptBudget
truncates retrieved chunks to fit within the model's context window,
preventing llama.cpp HTTP 400 errors.
"""

import logging
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.budget_utils import PromptBudget
from jet.adapters.llama_cpp.chunk_strategies import estimate_tokens_safe, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL

# Rich console for styled resource links
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
module_logger = logging.getLogger(__name__)

MODEL = LLM_MODEL
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer based only on the provided context chunks."
)
QUERY = "How does retrieval-augmented generation reduce hallucination?"
SOURCE_TEXT = (
    "Retrieval-augmented generation combines external knowledge with language models. "
    "It reduces hallucination by grounding responses in verified source material. "
    "The architecture typically consists of three components: a document ingestion pipeline "
    "that chunks and embeds text into a vector store, a retrieval module that selects "
    "the most relevant chunks based on semantic similarity to the user query, and a "
    "generation module that conditions the language model on both the query and the "
    "retrieved context to produce a faithful, cited answer. "
    "Chunking strategy directly impacts retrieval precision. "
    "Poorly chosen boundaries introduce noise that degrades answer quality. "
    "Token-based chunking guarantees chunks fit the embedding model and LLM budget. "
    "Overlap between chunks helps dense vector retrieval but adds indexing cost. "
    "Small-context models require aggressive reranking to 3-5 chunks before generation."
)
CHUNK_SIZE = 64
CHUNK_OVERLAP = 12
MIN_CHUNK_SIZE = 16
BUFFER = 4
MAX_COMPLETION_TOKENS = 256


def main() -> None:
    module_logger.info("=== Budget-Aware RAG Chunking Pipeline Demo ===")

    all_lines = []

    # Stage 1: Smart Chunking
    stage1_header = [
        f"\n{'=' * 60}",
        "📦 STAGE 1: Smart Chunking (retrieval_type='dense')",
        f"{'=' * 60}",
    ]
    print("\n".join(stage1_header))
    all_lines.extend(stage1_header)

    chunker = get_chunker("smart", model=MODEL)
    all_chunks = chunker.chunk(
        text=SOURCE_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
        retrieval_type="dense",
    )

    stage1_lines = [
        f"  Source: {len(SOURCE_TEXT)} chars, ~{estimate_tokens_safe(SOURCE_TEXT, MODEL)} tokens",
        f"  Chunks produced: {len(all_chunks)}",
    ]
    for line in stage1_lines:
        print(line)
    all_lines.extend(stage1_lines)

    for i, chunk in enumerate(all_chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        chunk_line = f"    [{i}] ({tok:>3d} tok) {chunk[:70]}..."
        print(chunk_line)
        all_lines.append(chunk_line)

    # Stage 2: Simulated Retrieval
    stage2_header = [
        f"\n{'=' * 60}",
        "🔎 STAGE 2: Simulated Retrieval (all chunks returned)",
        f"{'=' * 60}",
        f"  Retrieved: {len(all_chunks)} chunks",
    ]
    print("\n".join(stage2_header))
    all_lines.extend(stage2_header)

    # Stage 3: Budget Validation
    stage3_header = [
        f"\n{'=' * 60}",
        f"🔒 STAGE 3: Budget Validation (completion_reserve={MAX_COMPLETION_TOKENS})",
        f"{'=' * 60}",
    ]
    print("\n".join(stage3_header))
    all_lines.extend(stage3_header)

    budget = PromptBudget(MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS)
    safe_chunks = budget.validate(SYSTEM_PROMPT, QUERY, all_chunks)
    alloc = budget.get_allocation(SYSTEM_PROMPT, QUERY, all_chunks)

    stage3_lines = [
        f"  Model context:      {alloc.model_ctx} tokens",
        f"  System prompt:      {alloc.system_tokens} tokens",
        f"  Query:              {alloc.query_tokens} tokens",
        f"  Chat template OH:   {alloc.chat_template_overhead} tokens",
        f"  Completion reserve: {alloc.completion_reserve} tokens",
        f"  Available for chunks: {alloc.available_for_chunks} tokens",
        f"  Chunks included:    {alloc.chunks_included} / {len(all_chunks)}",
        f"  Chunks truncated:   {alloc.chunks_truncated}",
        f"  Within budget:      {'✅ YES' if alloc.within_budget else '❌ NO'}",
    ]
    for line in stage3_lines:
        print(line)
    all_lines.extend(stage3_lines)

    # Stage 4: Safe Prompt Assembly
    stage4_header = [
        f"\n{'=' * 60}",
        "✅ STAGE 4: Safe Prompt Assembly",
        f"{'=' * 60}",
        f"\n  System: {SYSTEM_PROMPT}",
        f"  User query: {QUERY}",
        f"  Context chunks ({len(safe_chunks)}):",
    ]
    print("\n".join(stage4_header))
    all_lines.extend(stage4_header)

    total_context_tokens = 0
    for i, chunk in enumerate(safe_chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        total_context_tokens += tok
        chunk_line = f"    [{i}] ({tok:>3d} tok) {chunk[:70]}..."
        print(chunk_line)
        all_lines.append(chunk_line)

    stage4_footer = [
        f"\n  Total context tokens: {total_context_tokens}",
        f"  Estimated total prompt: ~{alloc.total_used} / {alloc.model_ctx} tokens",
        f"  Remaining for completion: ~{alloc.model_ctx - alloc.total_used} tokens",
    ]
    for line in stage4_footer:
        print(line)
    all_lines.extend(stage4_footer)

    # Without Budget Validation
    without_header = [
        f"\n{'=' * 60}",
        "⚠️  WITHOUT BUDGET VALIDATION",
        f"{'=' * 60}",
    ]
    print("\n".join(without_header))
    all_lines.extend(without_header)

    naive_total = sum(estimate_tokens_safe(c, MODEL) for c in all_chunks)
    naive_prompt = (
        alloc.system_tokens
        + alloc.query_tokens
        + alloc.chat_template_overhead
        + naive_total
        + MAX_COMPLETION_TOKENS
    )

    without_lines = [
        f"  All {len(all_chunks)} chunks would use: {naive_total} tokens",
        f"  Total prompt would be: ~{naive_prompt} / {alloc.model_ctx} tokens",
    ]
    for line in without_lines:
        print(line)
    all_lines.extend(without_lines)

    if naive_prompt > alloc.model_ctx:
        over_msg = f"  ❌ EXCEEDS CONTEXT BY {naive_prompt - alloc.model_ctx} tokens → HTTP 400!"
        print(over_msg)
        all_lines.append(over_msg)
    else:
        fit_msg = (
            f"  ✅ Fits within budget (but may leave insufficient completion space)"
        )
        print(fit_msg)
        all_lines.append(fit_msg)

    print(f"\n{'=' * 60}")
    all_lines.append(f"\n{'=' * 60}")
    module_logger.info(
        "Demo complete. Budget validation guarantees safe llama.cpp requests."
    )

    # Save results
    summary_path = OUTPUT_DIR / "chunking_results.txt"
    summary_path.write_text("\n".join(all_lines), encoding="utf-8")
    console.print(
        f"💾 Results saved to [bold blue][link=file://{summary_path}]{summary_path.name}[/link][/bold blue]"
    )

    # Save all chunks (pre-budget)
    all_chunks_dir = OUTPUT_DIR / "chunks_all"
    all_chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(all_chunks):
        chunk_path = all_chunks_dir / f"chunk_{i:02d}.txt"
        tok = estimate_tokens_safe(chunk, MODEL)
        chunk_path.write_text(f"Tokens: {tok}\n\n{chunk}", encoding="utf-8")
    console.print(
        f"💾 All chunks saved to [bold blue][link=file://{all_chunks_dir}]{all_chunks_dir.name}/[/link][/bold blue] "
        f"({len(all_chunks)} files)"
    )

    # Save safe chunks (post-budget validation)
    safe_chunks_dir = OUTPUT_DIR / "chunks_safe"
    safe_chunks_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(safe_chunks):
        chunk_path = safe_chunks_dir / f"chunk_{i:02d}.txt"
        tok = estimate_tokens_safe(chunk, MODEL)
        chunk_path.write_text(f"Tokens: {tok}\n\n{chunk}", encoding="utf-8")
    console.print(
        f"💾 Safe chunks saved to [bold blue][link=file://{safe_chunks_dir}]{safe_chunks_dir.name}/[/link][/bold blue] "
        f"({len(safe_chunks)} files)"
    )

    # Save budget allocation details
    budget_path = OUTPUT_DIR / "budget_allocation.txt"
    budget_lines = [
        f"Model: {MODEL}",
        f"Model context: {alloc.model_ctx} tokens",
        f"System prompt tokens: {alloc.system_tokens}",
        f"Query tokens: {alloc.query_tokens}",
        f"Chat template overhead: {alloc.chat_template_overhead}",
        f"Completion reserve: {alloc.completion_reserve}",
        f"Available for chunks: {alloc.available_for_chunks}",
        f"Chunks included: {alloc.chunks_included} / {len(all_chunks)}",
        f"Chunks truncated: {alloc.chunks_truncated}",
        f"Total used: {alloc.total_used}",
        f"Within budget: {alloc.within_budget}",
    ]
    budget_path.write_text("\n".join(budget_lines), encoding="utf-8")
    console.print(
        f"💾 Budget allocation saved to [bold blue][link=file://{budget_path}]{budget_path.name}[/link][/bold blue]"
    )


if __name__ == "__main__":
    main()
