"""
Entry point for the Repo Agent CLI.
"""

import argparse
import asyncio

from agent import CodeAgent
from chunker import CodeChunker
from config import servers
from evaluator import evaluate_golden_dataset, print_eval_report, resolve_eval_repo
from jet_telemetry import initialize_telemetry
from scanner import RepositoryScanner
from utils import logger
from vector_store import VectorStore

initialize_telemetry(
    service_name="repo-agent-full",
    endpoint=servers.PHOENIX_URL
    if hasattr(servers, "PHOENIX_URL")
    else "http://localhost:6006",
)


async def run_indexing_pipeline(repo_path: str, force_reindex: bool = False):
    scanner = RepositoryScanner()
    chunker = CodeChunker()
    store = VectorStore()

    store.initialize()

    if force_reindex or not store.table.count_rows():
        logger.info(f"Starting full indexing for: {repo_path}")
        files = scanner.scan(repo_path)
        chunks = chunker.chunk_files(files)
        await store.add_chunks(chunks)
        logger.info(f"Indexing complete. Stored {len(chunks)} chunks.")
    else:
        logger.info("Using existing index. Use --reindex to update.")


async def main():
    parser = argparse.ArgumentParser(description="Local Repo Agent")
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Path to cloned repo (optional for --eval)",
    )
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    parser.add_argument("--eval", action="store_true", help="Run evaluation harness")
    args = parser.parse_args()

    # Resolve repo path: explicit arg OR golden repo
    resolved_repo = resolve_eval_repo(args.repo)
    repo_str = str(resolved_repo)

    if args.eval:
        # Index golden repo first (deterministic)
        await run_indexing_pipeline(repo_str, force_reindex=True)

        agent = CodeAgent(repo_path=repo_str)
        agent.load_repository_map()
        logger.info(" Starting Evaluation Harness...")
        eval_results = await evaluate_golden_dataset(agent, resolved_repo)
        print_eval_report(eval_results)
        return

    # Non-eval mode requires --repo explicitly
    if not args.repo:
        logger.error("Please provide a --repo path for interactive/query mode.")
        logger.info("Tip: Use --eval without --repo to test against the golden repo.")
        return

    await run_indexing_pipeline(repo_str, force_reindex=args.reindex)

    if args.query:
        agent = CodeAgent(repo_path=repo_str)
        agent.load_repository_map()
        print("\n Thinking...")
        result = await agent.run(args.query)
        print(f"\n💡 Answer ({result.get('intent', 'UNKNOWN')}):")
        print(result["answer"])
        if result["sources"]:
            print("\n📚 Sources:")
            for s in result["sources"]:
                print(f"  - {s['file']}")
    else:
        agent = CodeAgent(repo_path=repo_str)
        agent.load_repository_map()
        print("\n Repo Agent Ready (Agentic Mode). Type 'quit' to exit.")
        while True:
            try:
                q = input("\n❓ You: ")
                if q.lower() == "quit":
                    break
                if not q:
                    continue
                print("🤖 Thinking...")
                result = await agent.run(q)
                print(
                    f"\n Agent ({result.get('intent', 'UNKNOWN')}): {result['answer']}"
                )
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
