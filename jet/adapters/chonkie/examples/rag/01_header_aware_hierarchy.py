"""Header-aware hierarchical chunking for web-scraped content.

Preserves HTML heading hierarchy (h1→h2→h3) as chunk metadata so retrieval
can filter or boost results by section depth. Ideal for documentation sites,
Wikipedia-style articles, and any scraped content with structural headings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from jet.adapters.chonkie.llamacpp_embeddings import LlamacppEmbeddings
from jet.adapters.chonkie.llamacpp_genie import LlamacppGenie
from jet.logger import logger

# ---------------------------------------------------------------------------
# Simulated web-scraped HTML content
# ---------------------------------------------------------------------------

RAW_HTML = """\
<h1>Python AsyncIO Guide</h1>
<p>AsyncIO is Python's built-in library for asynchronous programming using
the async/await syntax. It enables concurrent execution without threads.</p>

<h2>Core Concepts</h2>
<p>An event loop is the central execution mechanism. It schedules coroutines
and manages I/O operations efficiently on a single thread.</p>

<h3>Coroutines</h3>
<p>Coroutines are functions defined with async def. They can be paused and
resumed, allowing other tasks to run during I/O waits.</p>

<h3>Futures and Tasks</h3>
<p>A Future represents a result that will be available later. A Task wraps
a coroutine and schedules it on the event loop automatically.</p>

<h2>Error Handling</h2>
<p>Exceptions in coroutines propagate like normal Python exceptions. Use
try/except inside async functions. Unhandled exceptions cancel the task.</p>

<h3>Timeout Patterns</h3>
<p>Use asyncio.wait_for() to set timeouts on coroutines. This prevents
indefinite blocking on external services or deadlocked operations.</p>

<h2>Performance Tips</h2>
<p>Avoid CPU-bound work in coroutines. Use asyncio.to_thread() or
ProcessPoolExecutor for parallel computation alongside async I/O.</p>
"""


@dataclass
class HierarchicalChunk:
    """Chunk enriched with heading hierarchy metadata."""

    text: str
    start_index: int
    end_index: int
    token_count: int
    h1: Optional[str] = None
    h2: Optional[str] = None
    h3: Optional[str] = None
    depth: int = 0  # 1=h1 level, 2=h2 level, 3=h3 level


# ---------------------------------------------------------------------------
# HTML hierarchy extractor
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"<(h[1-6])>(.*?)</\1>", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL | re.IGNORECASE)


def extract_hierarchical_sections(html: str) -> list[HierarchicalChunk]:
    """Parse HTML into sections tagged with their heading ancestry.

    Each paragraph inherits the most recent h1/h2/h3 context, producing
    flat chunks with full hierarchy metadata attached.
    """
    sections: list[HierarchicalChunk] = []
    current_h1: Optional[str] = None
    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    pos = 0

    # Walk through all headings and paragraphs in document order
    combined_pattern = re.compile(
        r"(<(?:h[1-6])>.*?</(?:h[1-6])>|<p>.*?</p>)",
        re.DOTALL | re.IGNORECASE,
    )

    for match in combined_pattern.finditer(html):
        tag_match = _HEADING_RE.match(match.group())
        para_match = _PARAGRAPH_RE.match(match.group())

        if tag_match:
            level = int(tag_match.group(1)[1])
            text = tag_match.group(2).strip()
            if level == 1:
                current_h1, current_h2, current_h3 = text, None, None
            elif level == 2:
                current_h2, current_h3 = text, None
            elif level == 3:
                current_h3 = text
        elif para_match:
            text = para_match.group(1).strip()
            depth = 3 if current_h3 else (2 if current_h2 else (1 if current_h1 else 0))
            sections.append(
                HierarchicalChunk(
                    text=text,
                    start_index=match.start(),
                    end_index=match.end(),
                    token_count=0,  # filled after embedding
                    h1=current_h1,
                    h2=current_h2,
                    h3=current_h3,
                    depth=depth,
                )
            )
        pos = match.end()

    return sections


# ---------------------------------------------------------------------------
# Simple metadata-aware vector store
# ---------------------------------------------------------------------------


class MetadataVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.metadata: list[dict] = []
        self.vectors: np.ndarray | None = None

    def add(self, texts: list[str], vectors: np.ndarray, meta: list[dict]) -> None:
        self.texts.extend(texts)
        self.metadata.extend(meta)
        self.vectors = (
            vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        )

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        meta_filter: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        if self.vectors is None:
            return []
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        sims = np.dot(self.vectors, query_vec) / norms

        # Apply metadata filter
        indices = list(range(len(self.texts)))
        if meta_filter:
            indices = [
                i
                for i in indices
                if all(self.metadata[i].get(k) == v for k, v in meta_filter.items())
            ]

        filtered_sims = sims[indices]
        top_idx = np.argsort(filtered_sims)[::-1][:top_k]
        return [
            (self.texts[indices[i]], float(filtered_sims[i]), self.metadata[indices[i]])
            for i in top_idx
        ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    embeddings = LlamacppEmbeddings(batch_size=64, show_progress=True)
    genie = LlamacppGenie(temperature=0.3)
    store = MetadataVectorStore()

    # 1. Extract hierarchical sections from HTML
    logger.info("=== Extracting Header Hierarchy ===")
    sections = extract_hierarchical_sections(RAW_HTML)
    logger.info(f"Extracted {len(sections)} sections with hierarchy tags")

    for s in sections:
        logger.debug(
            f"  [{s.depth}] h1={s.h1} | h2={s.h2} | h3={s.h3} | {s.text[:60]}..."
        )

    # 2. Embed sections and store with metadata
    logger.info("=== Embedding & Storing with Metadata ===")
    texts = [s.text for s in sections]
    vectors = np.array(embeddings.embed_batch(texts))
    meta = [{"h1": s.h1, "h2": s.h2, "h3": s.h3, "depth": s.depth} for s in sections]
    store.add(texts, vectors, meta)

    # 3. Query WITHOUT filter (global search)
    query = "How do timeouts work in async code?"
    logger.info(f"=== Global Search: '{query}' ===")
    q_vec = embeddings.embed(query)
    global_results = store.search(q_vec, top_k=3)
    print("\n🌍 Global Results:")
    for text, score, m in global_results:
        print(
            f"  Score: {score:.4f} | Depth: {m['depth']} | h2={m.get('h2')} | {text[:80]}..."
        )

    # 4. Query WITH header filter (section-scoped search)
    logger.info("=== Filtered Search: h2='Error Handling' ===")
    filtered_results = store.search(
        q_vec, top_k=3, meta_filter={"h2": "Error Handling"}
    )
    print("\n🎯 Filtered Results (h2='Error Handling'):")
    for text, score, m in filtered_results:
        print(f"  Score: {score:.4f} | h3={m.get('h3')} | {text[:80]}...")

    # 5. Generate answer with scoped context
    context = "\n\n".join(
        f"[{m.get('h2', '')}/{m.get('h3', '')}] {t}" for t, _, m in filtered_results
    )
    prompt = (
        f"Answer using ONLY this context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    answer = genie.generate(prompt)
    print(f"\n💡 Answer:\n{answer}")


if __name__ == "__main__":
    main()
