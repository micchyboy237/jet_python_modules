#!/usr/bin/env python3
"""
Example showing how to use cognee.start_ui() to launch the frontend.

This demonstrates the new UI functionality that works similar to DuckDB's start_ui().
"""

import asyncio
import cognee
import time
from dotenv import load_dotenv

load_dotenv("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/cognee/examples/.env")

import os
import shutil
from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.orange(f"Logs: {log_file}")


async def main():
    # First, let's add some data to cognee for the UI to display
    logger.debug("Adding sample data to cognee...")
    await cognee.add(
        "Natural language processing (NLP) is an interdisciplinary subfield of computer science and information retrieval."
    )
    await cognee.add(
        "Machine learning (ML) is a subset of artificial intelligence that focuses on algorithms and statistical models."
    )

    # Generate the knowledge graph
    logger.debug("Generating knowledge graph...")
    await cognee.cognify()

    logger.debug("\n" + "=" * 60)
    logger.debug("Starting cognee UI...")
    logger.debug("=" * 60)

    # Start the UI server
    server = cognee.start_ui(
        pid_callback=lambda pid: logger.debug(f"Started process with PID: {pid}"),
        host="localhost",
        port=3001,
        open_browser=True,
        auto_download=True,
        start_backend=True,
        backend_host="localhost",
        backend_port=8000,
    )

    if server:
        logger.debug("UI server started successfully!")
        logger.debug("The interface will be available at: http://localhost:3000")
        logger.debug("\nPress Ctrl+C to stop the server when you're done...")

        try:
            # Keep the server running
            while server.poll() is None:  # While process is still running
                time.sleep(1)
        except KeyboardInterrupt:
            logger.debug("\nStopping UI server...")
            server.terminate()
            server.wait()  # Wait for process to finish
            logger.debug("UI server stopped.")
    else:
        logger.debug("Failed to start UI server. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
