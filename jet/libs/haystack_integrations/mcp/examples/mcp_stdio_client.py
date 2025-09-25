# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging

from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo

from jet.logger import logger, CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file, level=logging.DEBUG)
logger.info(f"Logs: {log_file}")

# Setup logging
mcp_log_file = os.path.join(OUTPUT_DIR, "mcp.log")
mcp_logger = CustomLogger("haystack_integrations.tools.mcp", mcp_log_file)
mcp_logger.setLevel(logging.DEBUG)

# For stdio MCPTool we don't need to run a server, we can just use the MCPTool directly
# Here we use the mcp-server-time server
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time


def main():
    """Example of using the MCPTool implementation with stdio transport."""

    stdio_tool = None
    try:
        stdio_tool = MCPTool(
            name="get_current_time",
            server_info=StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"]),
        )

        logger.info(f"Tool spec: {stdio_tool.tool_spec}")

        result = stdio_tool.invoke(timezone="America/New_York")
        result_dict = json.loads(result)
        logger.success(f"Current time in New York: {result_dict["content"][0]["text"]}")

        result = stdio_tool.invoke(timezone="America/Los_Angeles")
        result_dict = json.loads(result)
        logger.success(f"Current time in Los Angeles: {result_dict["content"][0]["text"]}")
    except Exception as e:
        logger.error(f"Error in stdio example: {e}")
    finally:
        if stdio_tool:
            stdio_tool.close()


if __name__ == "__main__":
    main()
