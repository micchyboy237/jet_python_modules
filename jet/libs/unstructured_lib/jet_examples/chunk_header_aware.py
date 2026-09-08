import shutil
from pathlib import Path

from jet.file.utils import save_file
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.html import partition_html

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

url = "https://docs.unstructured.io/"
elements = partition_html(url=url)
chunks = chunk_by_title(elements, combine_text_under_n_chars=0)


def enrich_chunks_with_hierarchy(chunks: list) -> list:
    """Walk chunks sequentially, maintaining a heading stack to build breadcrumb titles."""
    # Stack entries: (depth, title_text)
    heading_stack: list[tuple[int, str]] = []

    for chunk in chunks:
        orig = getattr(chunk.metadata, "orig_elements", []) or []

        # Find the first Title element in this chunk
        title_elem = next(
            (e for e in orig if getattr(e, "category", None) == "Title"), None
        )

        if title_elem is not None:
            depth = getattr(title_elem.metadata, "category_depth", 0) or 0
            title_text = str(title_elem.text).strip()

            # Pop stack back to current depth (siblings/uncles removed)
            while heading_stack and heading_stack[-1][0] >= depth:
                heading_stack.pop()

            # Push current heading
            heading_stack.append((depth, title_text))

        # Build breadcrumb from current stack
        breadcrumb = " > ".join(text for _, text in heading_stack)

        # Store as ad-hoc metadata field (survives serialization)
        chunk.metadata.section_hierarchy = breadcrumb  # type: ignore[attr-defined]

    return chunks


chunks = enrich_chunks_with_hierarchy(chunks)

# Verify output
for i, chunk in enumerate(chunks):
    hierarchy = getattr(chunk.metadata, "section_hierarchy", "N/A")
    print(f"Chunk {i + 1}: {hierarchy}")

save_file(elements, OUTPUT_DIR / "elements.json")
save_file(chunks, OUTPUT_DIR / "chunks.json")
