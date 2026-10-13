"""
Entry point for the Repo Agent CLI.
"""

import argparse
import asyncio

from agent import CodeAgent
from chunker import CodeChunker
from config import servers
from evaluator import evaluate_golden_dataset, print_eval_report
from jet_telemetry import initialize_telemetry
from scanner import RepositoryScanner
from utils import logger
from vector_store import VectorStore

# 1. Initialize Telemetry ONCE at module load or start of main
# Do NOT call this inside loops or helper functions
initialize_telemetry(
    service_name="repo-agent-full",
    endpoint=servers.PHOENIX_URL
    if hasattr(servers, "PHOENIX_URL")
    else "http://localhost:6006",
)


@chain(name="indexing-pipeline")  # <--- ROOT SPAN: Ensures single Trace ID
async def run_indexing_pipeline(repo_path: str, force_reindex: bool = False):
    """
    Orchestrates the entire indexing process under ONE trace.
    All child spans (scan, chunk, embed) will automatically link here.
    """
    scanner = RepositoryScanner()
    chunker = CodeChunker()
    store = VectorStore()

    store.initialize()

    if force_reindex or not store.table.count_rows():
        logger.info(f"Starting full indexing for: {repo_path}")

        # These calls are now CHILDREN of 'indexing-pipeline'
        files = scanner.scan(repo_path)
        chunks = chunker.chunk_files(files)
        await store.add_chunks(chunks)

        logger.info(f"Indexing complete. Stored {len(chunks)} chunks.")
    else:
        logger.info("Using existing index. Use --reindex to update.")


async def main():
    parser = argparse.ArgumentParser(description="Local Repo Agent")
    parser.add_argument("--repo", type=str, help="Path to cloned repo")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    parser.add_argument("--eval", action="store_true", help="Run evaluation harness")
    args = parser.parse_args()

    if args.repo:
        # Index first
        await run_indexing_pipeline(args.repo, force_reindex=args.reindex)

        # Run evaluation if requested
        if args.eval:
            agent = CodeAgent(repo_path=args.repo)
            agent.load_repository_map()
            logger.info(" Starting Evaluation Harness...")
            eval_results = await evaluate_golden_dataset(agent, args.repo)
            print_eval_report(eval_results)
            return

        # Normal interactive/query mode
        if args.query:
            agent = CodeAgent(repo_path=args.repo)
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
            agent = CodeAgent(repo_path=args.repo)
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
                        f"\n💡 Agent ({result.get('intent', 'UNKNOWN')}): {result['answer']}"
                    )
                except KeyboardInterrupt:
                    break
    else:
        logger.error("Please provide a --repo path.")


if __name__ == "__main__":
    asyncio.run(main())
