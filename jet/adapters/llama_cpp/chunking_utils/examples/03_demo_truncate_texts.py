"""
Demo: Text Truncation
Shows how to truncate text to fit within model context windows.
"""

import logging

from jet.adapters.llama_cpp.chunking_utils import truncate_texts

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    long_text = (
        "The history of computing spans centuries of innovation. "
        "Early mechanical devices paved the way for electronic computers. "
        "The invention of the transistor revolutionized hardware design. "
        "Microprocessors enabled personal computing in the 1970s. "
        "Today, cloud computing powers global digital infrastructure. "
        "Quantum computing represents the next frontier in processing."
    )

    logger.info("Starting truncation demo...")

    # Truncate to a very small limit to demonstrate boundary handling
    truncated = truncate_texts(
        texts=long_text,
        max_tokens=35,  # Force truncation
        strict_sentences=True,  # Preserve sentence boundaries
        show_progress=True,
    )

    print("\n=== Original Length ===")
    print(f"{len(long_text)} chars")

    print("\n=== Truncated Result ===")
    print(truncated)
    print(f"{len(truncated)} chars")


if __name__ == "__main__":
    main()
