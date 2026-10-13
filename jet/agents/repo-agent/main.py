"""
Entry point for the Repo Agent CLI.
"""

import argparse
import asyncio

from chunker import CodeChunker
from query_engine import QueryEngine
from scanner import RepositoryScanner
from utils import logger
from vector_store import VectorStore


async def main():
    parser = argparse.ArgumentParser(description="Local Repo Agent")
    parser.add_argument("--repo", type=str, help="Path to cloned repo")
    parser.add_argument("--query", type=str, help="Question to ask")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing")
    args = parser.parse_args()

    engine = QueryEngine()

    if args.repo:
        logger.info(f"Processing repository: {args.repo}")
        scanner = RepositoryScanner()
        chunker = CodeChunker()
        store = VectorStore()

        # Initialize DB
        store.initialize()

        if args.reindex or not store.table.count_rows():
            files = scanner.scan(args.repo)
            chunks = chunker.chunk_files(files)
            await store.add_chunks(chunks)
        else:
            logger.info("Using existing index. Use --reindex to update.")

    if args.query:
        if not args.repo:
            logger.error("Please provide a --repo path to query.")
            return

        print("\n🤔 Thinking...")
        result = await engine.query(args.query)

        print("\n💡 Answer:")
        print(result["answer"])

        if result["sources"]:
            print("\n📚 Sources:")
            for s in result["sources"]:
                print(f"  - {s['file']} (Lines {s['lines']})")
    else:
        # Interactive Mode
        if not args.repo:
            logger.error("Interactive mode requires a --repo path.")
            return

        print("\n🤖 Repo Agent Ready. Type 'quit' to exit.")
        while True:
            try:
                q = input("\n❓ You: ")
                if q.lower() == "quit":
                    break
                if not q:
                    continue

                result = await engine.query(q)
                print(f"\n💡 Agent: {result['answer']}")
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    asyncio.run(main())
