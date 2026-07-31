# web_researcher/main.py
"""
Main entry point for the web researcher agent system.
"""

import logging
import sys
from pathlib import Path

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent))

from agents.manager import create_web_researcher
from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the web researcher with a sample query."""
    # Create the researcher
    researcher = create_web_researcher(
        llm_model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        max_context_length=4096,
        temperature=0.7,
        max_tokens=512,
        verbose=True,
    )

    # Sample query
    query = """
    What are the latest developments in quantum computing?
    Focus on:
    1. Recent breakthroughs (last 6 months)
    2. Key companies or research institutions involved
    3. Commercial applications being developed
    """

    logger.info("=" * 60)
    logger.info("Starting web research...")
    logger.info("=" * 60)

    try:
        result = researcher.run(query)
        logger.info("=" * 60)
        logger.info("RESEARCH COMPLETE")
        logger.info("=" * 60)
        print("\n" + "=" * 60)
        print("FINAL ANSWER:")
        print("=" * 60)
        print(result)
        print("\n" + "=" * 60)

        # Show token stats
        stats = researcher.get_token_stats()
        print(f"\nToken Usage:")
        print(f"  Input tokens:  {stats['total_input_tokens']}")
        print(f"  Output tokens: {stats['total_output_tokens']}")
        print(f"  Total tokens:  {stats['total_tokens']}")

    except KeyboardInterrupt:
        logger.info("Research interrupted by user")
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
