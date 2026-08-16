import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.orchestrator import AgenticOrchestrator
from tools.code_interpreter import CODE_SCHEMA, code_interpreter
from tools.registry import ToolRegistry
from tools.web_extractor import EXTRACTOR_SCHEMA, web_extractor
from tools.web_search import SEARCH_SCHEMA, web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("web_search", web_search, SEARCH_SCHEMA)
    reg.register("web_extractor", web_extractor, EXTRACTOR_SCHEMA)
    reg.register("code_interpreter", code_interpreter, CODE_SCHEMA)
    return reg


def main():
    registry = build_registry()
    agent = AgenticOrchestrator(registry)

    print("🤖 Qwen Studio Style Agent (Jet-Integrated)")
    print("=" * 50)

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit"):
            break
        print(f"\n{agent.run(query)}")


if __name__ == "__main__":
    main()
