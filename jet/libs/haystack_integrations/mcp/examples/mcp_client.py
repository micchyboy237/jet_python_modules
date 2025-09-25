# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging

from haystack_integrations.tools.mcp import MCPTool, SSEServerInfo, StreamableHttpServerInfo

from jet.logger import logger, CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

# Setup targeted logging - only show debug logs from our package
mcp_log_file = os.path.join(OUTPUT_DIR, "mcp.log")
mcp_logger = CustomLogger("haystack_integrations.tools.mcp", mcp_log_file)
mcp_logger.basicConfig(level=logging.WARNING)  # Set root logger to WARNING
mcp_logger.setLevel(logging.DEBUG)
# Ensure we have at least one handler to avoid messages going to root logger
if not mcp_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    mcp_logger.addHandler(handler)
    mcp_logger.propagate = False  # Prevent propagation to root logger

# Run this client after running the server mcp_server.py
# It shows how easy it is to use the MCPTool with different transport options


def main():
    """Example of MCPTool usage with server connection."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run an MCP client to connect to the server")
    parser.add_argument(
        "--transport",
        type=str,
        default="sse",
        choices=["sse", "streamable-http"],
        help="Transport mechanism for the MCP client (default: sse)",
    )
    args = parser.parse_args()

    # Construct the appropriate URL based on transport type
    if args.transport == "sse":
        server_info = SSEServerInfo(url="http://localhost:8000/sse")
    else:  # streamable-http
        server_info = StreamableHttpServerInfo(url="http://localhost:8000/mcp")
    tool = None
    tool_subtract = None
    try:
        tool = MCPTool(name="add", server_info=server_info)
        tool_subtract = MCPTool(name="subtract", server_info=server_info)

        result = tool.invoke(a=7, b=3)
        result_dict = json.loads(result)
        logger.success(f"7 + 3 = {result_dict["content"][0]["text"]}")

        result = tool_subtract.invoke(a=5, b=3)
        result_dict = json.loads(result)
        logger.success(f"5 - 3 = {result_dict["content"][0]["text"]}")

        result = tool.invoke(a=10, b=20)
        result_dict = json.loads(result)
        logger.success(f"10 + 20 = {result_dict["content"][0]["text"]}")

    except Exception as e:
        logger.error(f"Error in client example: {e}")
    finally:
        if tool:
            tool.close()
        if tool_subtract:
            tool_subtract.close()


if __name__ == "__main__":
    main()
