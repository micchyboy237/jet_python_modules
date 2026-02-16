# JetScripts/search/heuristics/run_generic_search.py
"""
Rich demo & test runner for GenericSearchEngine
"""

import logging
import sys
from typing import Any

from jet.search.heuristics.generic_search import (
    GenericSearchEngine,
    SearchResult,
    search_items,
)

# ────────────────────────────────────────────────
#   Logging Setup - colorful & detailed
# ────────────────────────────────────────────────


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: "%(asctime)s %(levelname)-6s %(message)s",
        logging.INFO: "%(asctime)s %(levelname)-6s %(message)s",
        logging.WARNING: "%(asctime)s %(levelname)-6s %(message)s",
        logging.ERROR: "%(asctime)s %(levelname)-6s %(message)s",
        logging.CRITICAL: "%(asctime)s %(levelname)-6s %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        color = {
            logging.DEBUG: self.grey,
            logging.INFO: self.green,
            logging.WARNING: self.yellow,
            logging.ERROR: self.red,
            logging.CRITICAL: self.bold_red,
        }.get(record.levelno, self.reset)

        formatter = logging.Formatter(
            f"{color}{log_fmt}{self.reset}", datefmt="%H:%M:%S"
        )
        return formatter.format(record)


def setup_logging(level=logging.DEBUG):
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    root.handlers = [handler]  # replace all handlers

    logging.info("Logging initialized with colored output")


# ────────────────────────────────────────────────
#   Sample Documents – very diverse
# ────────────────────────────────────────────────


def create_test_documents() -> list[dict[str, Any]]:
    return [
        {
            "id": "doc1",
            "title": "Python Performance Optimization Guide 2025",
            "content": """Python remains one of the most popular programming languages in 2025.
            However its interpreted nature and Global Interpreter Lock (GIL) still present performance challenges.
            This article explores modern techniques including asyncio, multiprocessing, Cython, Numba,
            PyPy, Mojo 🔥, and recent Python 3.13+ JIT experiments.""",
            "tags": "python performance optimization numba pypy asyncio",
            "author": "Dr. Elena Vargas",
            "date": "2025-11-12",
        },
        {
            "id": "doc2",
            "title": "Mojo Lang: The Future of Systems Programming?",
            "content": """Mojo, introduced by Modular in 2023, promises Python-like syntax with C-like performance.
            It features optional static typing, ownership model inspired by Rust, SIMD support,
            and direct GPU programming capabilities. Early benchmarks show 35,000× speedup over CPython
            in matrix multiplication. Is this the end of C++ dominance in high-performance computing?""",
            "tags": "mojo modular performance systems rust gpu",
            "author": "Marcus Chen",
            "date": "2025-08-03",
        },
        {
            "id": "doc3",
            "title": "Rust vs Go vs Zig in 2026 – Backend Services Comparison",
            "content": """We deployed three microservices (order processing) written in Rust, Go and Zig.
            Results after 4 weeks in production (50k req/s peak):
            • Rust: 11.2 μs p99, 1.4 GiB RSS, 3.1% CPU
            • Go:   18.7 μs p99, 2.8 GiB RSS, 4.9% CPU
            • Zig:  10.8 μs p99, 0.9 GiB RSS, 2.8% CPU
            Zig surprised everyone with lowest memory usage and excellent performance.""",
            "tags": "rust go zig performance backend microservices",
            "author": "Backend Team @ScaleX",
            "date": "2026-01-29",
        },
        {
            "id": "doc4",
            "title": "Почему Python всё ещё популярен в 2026 году? (Why Python is still popular in 2026)",
            "content": """Несмотря на критику за скорость выполнения, Python продолжает доминировать в:
            • Data Science & ML (pandas, polars, jax, torch 2.4+)
            • Автоматизация и скрипты
            • Обучение программированию
            • Прототипирование AI-систем
            Благодаря экосистеме и простоте он остаётся #1 языком по опросам StackOverflow 2025–2026.""",
            "tags": "python популярность data-science ml",
            "author": "Алексей Петров",
            "date": "2026-02-10",
        },
        {
            "id": "doc5",
            "title": "Short note – just numbers",
            "content": "2025 2026 3.14 2.718 1e-10 0xCAFEBABE 42",
            "tags": "numbers math constants",
            "author": "Calculator Bot",
            "date": "2026-02-15",
        },
        {
            "id": "doc6",
            "title": "",
            "content": "   ",
            "tags": "",
            "author": "Empty",
            "date": "1970-01-01",
        },
    ]


def doc_text_extractor(doc: dict) -> dict[str, str]:
    """How we flatten document into searchable fields"""
    return {
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "tags": doc.get("tags", ""),
        "author": doc.get("author", ""),
    }


# ────────────────────────────────────────────────
#   Pretty printing helpers
# ────────────────────────────────────────────────


def print_result(res: SearchResult[dict], rank: int):
    item = res.item
    print(f"\n{rank}. {item['title'] or '(no title)'}")
    print(f"   score = {res.score:.4f}")
    print(f"   matched fields: {', '.join(res.matched_fields)}")
    print(f"   matched terms : {res.matched_terms}")

    if res.highlights:
        print("   highlights:")
        for field, text in res.highlights.items():
            # Shorten very long highlights
            if len(text) > 140:
                text = text[:137] + "..."
            print(f"     {field:10} | {text}")


def run_demo_search(
    engine_or_items,
    query: str,
    logic: str = "AND",
    limit: int = 6,
    weights: dict | None = None,
    name: str = "unnamed",
):
    print(f"\n{'═' * 70}")
    print(f"► {name.upper()}")
    print(f"  query : {query!r}")
    print(f"  logic : {logic}")
    print(f"  weights: {weights}")
    print(f"{'═' * 70}")

    if isinstance(engine_or_items, GenericSearchEngine):
        results = engine_or_items.search(
            query=query,
            logic=logic,  # type: ignore
            limit=limit,
        )
    else:
        results = search_items(
            items=engine_or_items,
            query=query,
            text_extractor=doc_text_extractor,
            field_weights=weights,
            logic=logic,  # type: ignore
            limit=limit,
        )

    if not results:
        print("  (no matches)")
        return

    for i, r in enumerate(results, 1):
        print_result(r, i)


# ────────────────────────────────────────────────
#   Main Demo
# ────────────────────────────────────────────────


def main():
    setup_logging()

    docs = create_test_documents()
    logging.info(f"Loaded {len(docs)} test documents")

    # ── Variant 1: Default weights ───────────────────────────────────
    print("\n" + "═" * 100)
    print(" DEFAULT WEIGHTS (title & content = 1.0)")
    engine = GenericSearchEngine(docs, doc_text_extractor)

    run_demo_search(engine, "python performance", name="basic python perf")
    run_demo_search(engine, "mojo rust zig", logic="OR", name="systems lang OR")
    run_demo_search(engine, "python 2026", name="python + year")

    # ── Variant 2: Boost title & tags ────────────────────────────────
    boosted_weights = {
        "title": 2.5,
        "tags": 2.0,
        "content": 1.0,
        "author": 0.6,
    }

    run_demo_search(
        docs,
        "rust backend",
        logic="AND",
        weights=boosted_weights,
        name="boosted title+tags – rust backend",
    )

    run_demo_search(
        docs,
        "python",
        logic="OR",
        weights=boosted_weights,
        name="python – title/tags boosted",
    )

    # ── Variant 3: Unicode + edge cases ──────────────────────────────
    run_demo_search(docs, "почему python", name="unicode query (russian)")

    run_demo_search(docs, "42 3.14", name="numbers search")

    run_demo_search(docs, "   ", name="empty query")

    run_demo_search(docs, "supercalifragilisticexpialidocious", name="impossible word")

    logging.info("Demo completed")


if __name__ == "__main__":
    main()
