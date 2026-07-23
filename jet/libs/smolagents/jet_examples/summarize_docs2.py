import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import DirectoryLoader, MarkdownLoader
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ========================= CONFIG =========================
TARGET_DIR = "./demo_docs"  # Change as needed
MODEL_PATH = "/path/to/your/model.gguf"  # e.g., Phi-3-mini-4k-instruct.Q5_K_M.gguf
N_GPU_LAYERS = 35  # Adjust for GTX 1660 (higher = faster, watch VRAM)
N_CTX = 10000
MAX_CONCURRENT = 3  # Safe for your Ryzen setup; increase if stable
CHUNK_SIZE = 3500  # Characters (~tokens)
CHUNK_OVERLAP = 300
SUMMARY_MAX_TOKENS = 800

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("summarization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ========================= PROMPTS (Agent Instructions) =========================
MAP_PROMPT = PromptTemplate.from_template(
    """You are a precise Summarizer Agent. 
    Provide a concise, structured bullet-point summary of the Markdown content.
    Focus on: key concepts, structure (headings), decisions, and actionable insights.
    Output only bullets. Keep under {max_tokens} words.

    Content:
    {context}"""
)

REDUCE_PROMPT = PromptTemplate.from_template(
    """You are the Aggregator Agent. Synthesize multiple summaries into a coherent overview.
    Identify main themes, connections across sections/files, and overall insights.
    Structure output with: ## Overall Summary, ### Key Themes, ### Cross-References.

    Summaries to synthesize:
    {docs}"""
)

VERIFY_PROMPT = PromptTemplate.from_template(
    """You are the Verifier Agent. Review the final summary for accuracy, completeness, and hallucinations.
    Suggest improvements if needed. Output: APPROVED or REVISE + reasons.

    Summary:
    {summary}

    Original context samples (first 3):
    {samples}"""
)


# ========================= LLM SETUP =========================
def init_llm() -> LlamaCpp:
    return LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_batch=512,
        n_ctx=N_CTX,
        temperature=0.2,
        max_tokens=SUMMARY_MAX_TOKENS,
        verbose=False,
        streaming=False,
    )


llm = init_llm()


# ========================= DEMO SETUP =========================
def create_demo_docs():
    """Create sample Markdown files for demo if directory empty."""
    os.makedirs(TARGET_DIR, exist_ok=True)
    if any(Path(TARGET_DIR).glob("**/*.md")):
        return

    logger.info("Creating demo Markdown files...")
    samples = {
        "project_overview.md": "# Project Overview\n\nThis is a demo project for AI summarization.\n\n## Goals\n- Summarize docs recursively\n- Handle token limits",
        "architecture.md": "# System Architecture\n\n## Components\n1. Explorer Agent\n2. Parallel Summarizers\n\n**Key Decision**: Use hierarchical reduce.",
        "api_specs.md": "# API Endpoints\n\n## GET /summarize\n- Input: directory path\n- Output: hierarchical summary",
        "subdir/notes.md": "# Implementation Notes\n\nUse asyncio for parallelism. Monitor GPU usage.",
    }
    for fname, content in samples.items():
        path = Path(TARGET_DIR) / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    logger.info(f"Demo docs created in {TARGET_DIR}")


# ========================= CORE AGENTS =========================
async def explorer_agent() -> List[Document]:
    """Explorer Agent: Recursively load all Markdown files."""
    logger.info(f"Exploring directory: {TARGET_DIR}")
    loader = DirectoryLoader(
        TARGET_DIR,
        glob="**/*.md",
        loader_cls=MarkdownLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        use_multithreading=True,
    )
    docs = loader.load()
    logger.info(f"Explorer found {len(docs)} Markdown documents")
    for doc in docs[:5]:  # Log sample metadata
        logger.info(f"  - {doc.metadata.get('source')}")
    return docs


def chunker_agent(docs: List[Document]) -> List[Document]:
    """Chunker Agent: Token-aware splitting with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "##", "#", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Chunker created {len(chunks)} chunks")
    return chunks


async def summarizer_agent(chunk: Document, semaphore: asyncio.Semaphore) -> Dict:
    """Summarizer Agent (parallelizable)."""
    async with semaphore:
        try:
            prompt = MAP_PROMPT.format(
                context=chunk.page_content, max_tokens=SUMMARY_MAX_TOKENS // 2
            )
            summary = await asyncio.to_thread(llm.invoke, prompt)
            result = {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_id": id(chunk),
                "summary": str(summary),
                "tokens_approx": len(str(summary).split()),
            }
            logger.info(f"Summarizer completed: {result['source']}")
            return result
        except Exception as e:
            logger.error(f"Summarizer failed for {chunk.metadata.get('source')}: {e}")
            return {"source": chunk.metadata.get("source"), "error": str(e)}


async def map_phase(chunks: List[Document]) -> List[Dict]:
    """Map Phase: Parallel Summarizer Agents."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [summarizer_agent(chunk, semaphore) for chunk in chunks]
    results = []
    for f in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Map Phase (Summarizers)"
    ):
        results.append(await f)
    return [r for r in results if "error" not in r]


def hierarchical_reduce(summaries: List[Dict], level: int = 0) -> str:
    """Reduce Phase: Hierarchical collapse if needed."""
    if not summaries:
        return "No content to summarize."

    combined = "\n\n---\n\n".join([s["summary"] for s in summaries])
    token_est = len(combined.split())  # rough

    if token_est > (N_CTX * 0.6) and len(summaries) > 5:
        # Hierarchical collapse: group into batches
        logger.info(f"Level {level}: Collapsing {len(summaries)} summaries (too large)")
        batch_size = max(3, len(summaries) // 3)
        batches = [
            summaries[i : i + batch_size] for i in range(0, len(summaries), batch_size)
        ]
        collapsed = []
        for batch in batches:
            batch_text = "\n\n".join([s["summary"] for s in batch])
            prompt = REDUCE_PROMPT.format(docs=batch_text)
            reduced = llm.invoke(prompt)
            collapsed.append({"summary": str(reduced)})
        return hierarchical_reduce(collapsed, level + 1)

    # Final reduce
    prompt = REDUCE_PROMPT.format(docs=combined)
    final_summary = llm.invoke(prompt)
    logger.info(f"Hierarchical Reduce completed at level {level}")
    return str(final_summary)


async def verifier_agent(final_summary: str, sample_summaries: List[Dict]) -> str:
    """Verifier Agent: Quality check."""
    samples = "\n\n".join([s["summary"][:300] for s in sample_summaries[:3]])
    prompt = VERIFY_PROMPT.format(summary=final_summary, samples=samples)
    verification = llm.invoke(prompt)
    logger.info(f"Verifier: {str(verification)[:200]}...")
    return str(verification)


# ========================= MAIN PIPELINE =========================
async def main():
    create_demo_docs()

    # Agent 1: Explorer
    docs = await explorer_agent()

    # Agent 2: Chunker
    chunks = chunker_agent(docs)

    # Agent 3+: Parallel Summarizers (Map)
    map_results = await map_phase(chunks)

    # Save intermediates for traceability
    with open("intermediate_summaries.json", "w", encoding="utf-8") as f:
        json.dump(map_results, f, indent=2, ensure_ascii=False)

    # Agent: Aggregator (Hierarchical Reduce)
    final_summary = hierarchical_reduce(map_results)

    # Agent: Verifier
    verification = await verifier_agent(final_summary, map_results)

    # Output
    output_path = "final_summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Final Hierarchical Summary\n\n")
        f.write(final_summary)
        f.write("\n\n## Verification Report\n\n")
        f.write(verification)
        f.write("\n\n## Metadata\n")
        f.write(f"- Total files processed: {len(docs)}\n")
        f.write(f"- Total chunks: {len(chunks)}\n")
        f.write(f"- Map summaries: {len(map_results)}\n")

    logger.info(f"✅ Complete! Final summary saved to {output_path}")
    print(
        f"\nFinal summary written to {output_path}. Check summarization.log and intermediate_summaries.json for details."
    )


if __name__ == "__main__":
    asyncio.run(main())
