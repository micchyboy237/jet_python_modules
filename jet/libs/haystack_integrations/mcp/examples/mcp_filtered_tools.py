# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from haystack_integrations.tools.mcp import MCPToolset, SSEServerInfo

from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

# This example demonstrates using MCPToolset with SSE transport
# and filtering tools by name
# Run this client after running the server mcp_server.py with sse transport
# It shows how MCPToolset can selectively include only specific tools


def main():
    """Example of using MCPToolset with filtered tools."""

    full_toolset = None
    filtered_toolset = None
    try:
        logger.gray("Creating toolset with all available tools:")
        # Create a toolset with all available tools
        full_toolset = MCPToolset(
            server_info=SSEServerInfo(url="http://localhost:8000/sse"),
        )

        # Print all discovered tools
        logger.info(f"Discovered {len(full_toolset)} tools:")
        for tool in full_toolset:
            logger.info(f"  - {tool.name}: {tool.description}")

        logger.gray("\nCreating toolset with filtered tools:")
        # Create a toolset with only specific tools
        # In this example, we're only including the 'add' tool
        filtered_toolset = MCPToolset(
            server_info=SSEServerInfo(url="http://localhost:8000/sse"),
            tool_names=["add"],  # Only include the 'add' tool
        )

        # Print filtered tools
        logger.info(f"Filtered toolset has {len(filtered_toolset)} tools:")
        for tool in filtered_toolset:
            logger.info(f"  - {tool.name}: {tool.description}")

        # Use the filtered toolset
        if len(filtered_toolset) > 0:
            add_tool = filtered_toolset.tools[0]  # The only tool should be 'add'
            result = add_tool.invoke(a=10, b=5)
            result_dict = json.loads(result)
            logger.success(f"\nInvoking {add_tool.name}: 10 + 5 = {result_dict["content"][0]["text"]}")
        else:
            logger.warning("No tools available in the filtered toolset")

    except Exception as e:
        logger.error(f"Error in filtered toolset example: {e}")
    finally:
        if full_toolset:
            full_toolset.close()
        if filtered_toolset:
            filtered_toolset.close()


if __name__ == "__main__":
    main()
