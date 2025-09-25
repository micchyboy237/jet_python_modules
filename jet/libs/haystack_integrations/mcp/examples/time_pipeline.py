# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Full example of a pipeline that uses MCPTool to get the current time
# and then uses the time to answer a user question.
# Here we use the mcp-server-time mcp package
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time

import logging

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
# from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp.mcp_tool import MCPTool, StdioServerInfo

import os
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.logger import logger, CustomLogger
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


def main():
    time_tool = None
    try:
        time_tool = MCPTool(
            name="get_current_time",
            server_info=StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"]),
        )
        pipeline = Pipeline()
        pipeline.add_component("llm", OllamaChatGenerator(model="qwen3:4b-q4_K_M", tools=[time_tool], agent_name="tool_llm"))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[time_tool]))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OllamaChatGenerator(model="qwen3:4b-q4_K_M", agent_name="response_llm"))
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is the time in New York? Be brief."  # can be any city
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        print(result["response_llm"]["replies"][0].text)
    finally:
        if time_tool:
            time_tool.close()


if __name__ == "__main__":
    main()
