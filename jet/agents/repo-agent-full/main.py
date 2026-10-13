"""
Entry point for the Repo Agent CLI.
"""

import argparse
import asyncio

from agent import CodeAgent
from chunker import CodeChunker
from scanner import RepositoryScanner
from utils import logger
from vector_store import VectorStore


async def main():
    parser = argparse.ArgumentParser(description="Local Repo Agent")
    parser.add_argument("--repo", type=str, help="Path to cloned repo")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    args = parser.parse_args()

    # Initialize components
    scanner = RepositoryScanner()
    chunker = CodeChunker()
    store = VectorStore()

    # Indexing Logic
    if args.repo:
        logger.info(f"Processing repository: {args.repo}")
        store.initialize()

        if args.reindex or not store.table.count_rows():
            files = scanner.scan(args.repo)
            chunks = chunker.chunk_files(files)
            await store.add_chunks(chunks)
        else:
            logger.info("Using existing index. Use --reindex to update.")

    # Querying Logic
    if args.query:
        if not args.repo:
            logger.error("Please provide a --repo path to query.")
            return

        # Use the new Agent
        agent = CodeAgent(repo_path=args.repo)
        agent.load_repository_map()  # Generate the map

        print("\n🤔 Thinking...")
        result = await agent.run(args.query)

        print(f"\n💡 Answer ({result.get('intent', 'UNKNOWN')}):")
        print(result["answer"])

        if result["sources"]:
            print("\n📚 Sources:")
            for s in result["sources"]:
                print(f"  - {s['file']}")

    else:
        # Interactive Mode
        if not args.repo:
            logger.error("Interactive mode requires a --repo path.")
            return

        agent = CodeAgent(repo_path=args.repo)
        agent.load_repository_map()
        print("\n🤖 Repo Agent Ready (Agentic Mode). Type 'quit' to exit.")
        while True:
            try:
                q = input("\n❓ You: ")
                if q.lower() == "quit":
                    break
                if not q:
                    continue

                print("🤔 Thinking...")
                result = await agent.run(q)
                print(
                    f"\n💡 Agent ({result.get('intent', 'UNKNOWN')}): {result['answer']}"
                )
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
