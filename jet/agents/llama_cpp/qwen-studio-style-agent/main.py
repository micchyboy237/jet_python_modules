import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.orchestrator import AgenticOrchestrator
from agent.search_manager import SEARCH_MANAGER_SCHEMA, search_manager
from tools.code_interpreter import CODE_SCHEMA, code_interpreter
from tools.registry import ToolRegistry
from tools.web_extractor import EXTRACTOR_SCHEMA, web_extractor
from tools.web_search import SEARCH_SCHEMA, web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry(blacklist_ttl_seconds=300)

    # Register search_manager FIRST — model prefers it for factual queries
    # Raw web_search/web_extractor remain available as fallbacks
    reg.register("search_manager", search_manager, SEARCH_MANAGER_SCHEMA)
    reg.register("web_search", web_search, SEARCH_SCHEMA)
    reg.register("web_extractor", web_extractor, EXTRACTOR_SCHEMA)
    reg.register("code_interpreter", code_interpreter, CODE_SCHEMA)

    return reg


def main():
    registry = build_registry()
    agent = AgenticOrchestrator(registry)

    print("🤖 Qwen Studio Style Agent (Failure-Resilient Verification)")
    print("=" * 55)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break

        # Optional: clear failure tracking between independent queries
        # registry.clear_session_failures()

        print(f"\n{agent.run(query)}")


if __name__ == "__main__":
    main()
