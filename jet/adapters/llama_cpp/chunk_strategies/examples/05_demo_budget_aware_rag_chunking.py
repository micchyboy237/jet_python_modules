# jet_python_modules/jet/adapters/llama_cpp/chunk_strategies/examples/08_demo_budget_aware_rag_chunking.py
"""Demo: Budget-aware RAG chunking pipeline.

Demonstrates end-to-end safe RAG: chunking → retrieval-type overlap →
budget validation → guaranteed-safe prompt assembly. Shows how PromptBudget
truncates retrieved chunks to fit within the model's context window,
preventing llama.cpp HTTP 400 errors.
"""

import logging

from jet.adapters.llama_cpp.budget_utils import PromptBudget
from jet.adapters.llama_cpp.chunk_strategies import estimate_tokens_safe, get_chunker
from jet.adapters.llama_cpp.config import LLM_MODEL

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

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
    logger.info("=== Budget-Aware RAG Chunking Pipeline Demo ===")

    # ── Stage 1: Chunk with retrieval-type awareness ──────────────────
    print(f"\n{'=' * 60}")
    print("📦 STAGE 1: Smart Chunking (retrieval_type='dense')")
    print(f"{'=' * 60}")

    chunker = get_chunker("smart", model=MODEL)
    all_chunks = chunker.chunk(
        text=SOURCE_TEXT,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
        buffer=BUFFER,
        retrieval_type="dense",
    )

    print(
        f"  Source: {len(SOURCE_TEXT)} chars, ~{estimate_tokens_safe(SOURCE_TEXT, MODEL)} tokens"
    )
    print(f"  Chunks produced: {len(all_chunks)}")
    for i, chunk in enumerate(all_chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        print(f"    [{i}] ({tok:>3d} tok) {chunk[:70]}...")

    # ── Stage 2: Simulate retrieval (all chunks "retrieved") ──────────
    print(f"\n{'=' * 60}")
    print("🔎 STAGE 2: Simulated Retrieval (all chunks returned)")
    print(f"{'=' * 60}")
    retrieved = all_chunks  # In real pipeline, this comes from hybrid_search
    print(f"  Retrieved: {len(retrieved)} chunks")

    # ── Stage 3: Budget validation ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"🔒 STAGE 3: Budget Validation (completion_reserve={MAX_COMPLETION_TOKENS})")
    print(f"{'=' * 60}")

    budget = PromptBudget(MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS)
    safe_chunks = budget.validate(SYSTEM_PROMPT, QUERY, retrieved)

    alloc = budget.get_allocation(SYSTEM_PROMPT, QUERY, retrieved)
    print(f"  Model context:      {alloc.model_ctx} tokens")
    print(f"  System prompt:      {alloc.system_tokens} tokens")
    print(f"  Query:              {alloc.query_tokens} tokens")
    print(f"  Chat template OH:   {alloc.chat_template_overhead} tokens")
    print(f"  Completion reserve: {alloc.completion_reserve} tokens")
    print(f"  Available for chunks: {alloc.available_for_chunks} tokens")
    print(f"  Chunks included:    {alloc.chunks_included} / {len(retrieved)}")
    print(f"  Chunks truncated:   {alloc.chunks_truncated}")
    print(f"  Within budget:      {'✅ YES' if alloc.within_budget else '❌ NO'}")

    # ── Stage 4: Safe prompt assembly ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("✅ STAGE 4: Safe Prompt Assembly")
    print(f"{'=' * 60}")

    print(f"\n  System: {SYSTEM_PROMPT}")
    print(f"  User query: {QUERY}")
    print(f"  Context chunks ({len(safe_chunks)}):")
    total_context_tokens = 0
    for i, chunk in enumerate(safe_chunks):
        tok = estimate_tokens_safe(chunk, MODEL)
        total_context_tokens += tok
        print(f"    [{i}] ({tok:>3d} tok) {chunk[:70]}...")

    print(f"\n  Total context tokens: {total_context_tokens}")
    print(f"  Estimated total prompt: ~{alloc.total_used} / {alloc.model_ctx} tokens")
    print(f"  Remaining for completion: ~{alloc.model_ctx - alloc.total_used} tokens")

    # ── Comparison: What happens without budget validation ────────────
    print(f"\n{'=' * 60}")
    print("⚠️  WITHOUT BUDGET VALIDATION")
    print(f"{'=' * 60}")

    naive_total = sum(estimate_tokens_safe(c, MODEL) for c in retrieved)
    naive_prompt = (
        alloc.system_tokens
        + alloc.query_tokens
        + alloc.chat_template_overhead
        + naive_total
        + MAX_COMPLETION_TOKENS
    )
    print(f"  All {len(retrieved)} chunks would use: {naive_total} tokens")
    print(f"  Total prompt would be: ~{naive_prompt} / {alloc.model_ctx} tokens")
    if naive_prompt > alloc.model_ctx:
        print(
            f"  ❌ EXCEEDS CONTEXT BY {naive_prompt - alloc.model_ctx} tokens → HTTP 400!"
        )
    else:
        print(f"  ✅ Fits within budget (but may leave insufficient completion space)")

    print(f"\n{'=' * 60}")
    logger.info("Demo complete. Budget validation guarantees safe llama.cpp requests.")


if __name__ == "__main__":
    main()
