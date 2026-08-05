import json
import shutil
from pathlib import Path

from jet.adapters.unstructured.document_parser import logger, parse_document

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

# Clean and recreate output directory
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Convert a file path/URL to a safe directory name."""
    # Remove protocol prefix
    name = name.replace("https://", "").replace("http://", "")
    # Replace path separators and special chars
    name = name.replace("/", "_").replace("\\", "_")
    name = name.replace(":", "_").replace("?", "_").replace("&", "_")
    # Limit length
    if len(name) > 80:
        name = name[:40] + "..." + name[-37:]
    return name


def save_result(result: dict, output_dir: Path, label: str) -> None:
    """
    Save all parse result artifacts to a subdirectory.

    Creates:
      {output_dir}/{sanitized_label}/
        ├── summary.json      - Parse metadata (no elements/chunks to keep it light)
        ├── rag_context.md    - Full RAG-ready text as markdown
        ├── chunks.json       - List of chunk dicts with text, token_count, strategy
        └── elements.json     - Raw element dicts from the parser
    """
    safe_name = sanitize_filename(label)
    doc_dir = output_dir / safe_name
    doc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary (lightweight metadata only — no heavy lists)
    summary = {
        "path": result["path"],
        "status": result["status"],
        "element_count": result["element_count"],
        "categories": result["categories"],
        "word_count": result["word_count"],
        "page_count": result["page_count"],
        "chunk_count": len(result.get("chunks", [])),
        "total_chunk_tokens": sum(
            c.get("token_count", 0) for c in result.get("chunks", [])
        ),
    }
    summary_path = doc_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"save_result | summary → {summary_path}")

    # 2. RAG context as markdown
    if result.get("rag_context"):
        rag_path = doc_dir / "rag_context.md"
        with open(rag_path, "w", encoding="utf-8") as f:
            f.write(result["rag_context"])
        logger.info(
            f"save_result | rag_context → {rag_path} "
            f"({len(result['rag_context'])} chars)"
        )

    # 3. Chunks (structured with strategies — useful for RAG retrieval debugging)
    if result.get("chunks"):
        chunks_path = doc_dir / "chunks.json"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(result["chunks"], f, indent=2, ensure_ascii=False)
        logger.info(
            f"save_result | chunks → {chunks_path} ({len(result['chunks'])} chunks)"
        )

    # 4. Raw elements (for debugging parser behavior)
    if result.get("elements"):
        elements_path = doc_dir / "elements.json"
        with open(elements_path, "w", encoding="utf-8") as f:
            json.dump(result["elements"], f, indent=2, ensure_ascii=False)
        logger.info(
            f"save_result | elements → {elements_path} ({len(result['elements'])} elements)"
        )


if __name__ == "__main__":
    test_inputs = [
        "/Users/jethroestrada/Downloads/Resume Latest - Jethro Estrada.pdf",
        "https://www.iana.org/help/example-domains",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.html",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.py",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.ipynb",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.log",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.txt",
    ]

    logger.info("=" * 70)
    logger.info("BATCH START")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 70)

    results = []
    for path in test_inputs:
        result = parse_document(path)
        label = path.split("/")[-1] if "/" in path else path
        status_icon = "✅" if result["status"] == "success" else "❌"
        n_chunks = len(result.get("chunks", []))
        total_chunk_tokens = sum(
            c.get("token_count", 0) for c in result.get("chunks", [])
        )

        # Save to disk if successful
        if result["status"] == "success":
            save_result(result, OUTPUT_DIR, label)

        summary = (
            f"{label:20s} | {status_icon} elems={result['element_count']:4d} | "
            f"chunks={n_chunks:3d} | tokens={total_chunk_tokens:5d} | "
            f"cats={result['categories']}"
        )
        results.append(summary)
        print(summary)

    # Also save batch summary
    batch_summary = {
        "total_inputs": len(test_inputs),
        "succeeded": sum(1 for r in results if "✅" in r),
        "failed": sum(1 for r in results if "❌" in r),
        "details": results,
    }
    batch_path = OUTPUT_DIR / "_batch_summary.json"
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Batch summary saved → {batch_path}")

    succeeded = batch_summary["succeeded"]
    logger.info("=" * 70)
    logger.info(f"BATCH COMPLETE | {succeeded}/{len(results)} succeeded")
    for line in results:
        logger.info(line)
    logger.info("=" * 70)
