import shutil
from pathlib import Path

from jet.adapters.unstructured.document_parser import logger, parse_document

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_inputs = [
        "/Users/jethroestrada/Downloads/Resume Latest - Jethro Estrada.pdf",
        "https://example.com",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.html",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.py",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.ipynb",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.log",
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.txt",
    ]

    logger.info("=" * 70)
    logger.info("BATCH START")
    logger.info("=" * 70)

    results = []
    for path in test_inputs:
        result = parse_document(path)
        label = path.split("/")[-1] if "/" in path else path
        status_icon = "✅" if result["status"] == "success" else "❌"
        n_chunks = len(result.get("chunks", []))
        total_chunk_tokens = sum(c["token_count"] for c in result.get("chunks", []))
        summary = (
            f"{label:20s} | {status_icon} elems={result['element_count']:4d} | "
            f"chunks={n_chunks:3d} | tokens={total_chunk_tokens:5d} | "
            f"cats={result['categories']}"
        )
        results.append(summary)
        print(summary)

    succeeded = sum(1 for r in results if "✅" in r)
    logger.info("=" * 70)
    logger.info(f"BATCH COMPLETE | {succeeded}/{len(results)} succeeded")
    for line in results:
        logger.info(line)
    logger.info("=" * 70)
