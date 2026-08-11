"""WebSwarm CLI entry point.

Run directly:  python swarm.py "your query"
Run as module: python -m webswarm.swarm "your query"
"""

import argparse
import asyncio
import logging
import os
import sys

# ── Bootstrap: ensure package imports work regardless of invocation method ──
# When run as `python swarm.py`, __package__ is None and relative imports fail.
# We add the parent directory to sys.path and set __package__ so that both
# `from .graph import ...` (relative) and absolute imports resolve correctly.
if __name__ == "__main__" and __package__ is None:
    _parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    __package__ = "webswarm"

# ── Logging setup (before any project imports) ──
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# ── Project imports (config sets CWD + creates generated/ on import) ──
from .config import _THIS_DIR  # noqa: F401 – triggers CWD + generated/ setup
from .graph import run_swarm


def main():
    parser = argparse.ArgumentParser(description="Run WebSwarm with a query.")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the supply chain risks for solid-state batteries in SE Asia?",
        help="The query to run WebSwarm on.",
    )
    args = parser.parse_args()
    answer = asyncio.run(run_swarm(args.query))
    print("\n" + "=" * 80 + "\n" + answer)


if __name__ == "__main__":
    main()
