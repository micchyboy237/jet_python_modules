# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Full example of a pipeline that uses MCPToolset to get the current time
# and then uses the time to answer a user question.
# Here we use the mcp-server-time mcp package
# See https://github.com/modelcontextprotocol/servers/tree/main/src/time for more details
# prior to running this script, pip install mcp-server-time

import os

from haystack import Pipeline
from haystack.components.converters import OutputAdapter
# from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage

from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo

from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.logger import logger
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")


def main():
    # Create server info for the time service
    server_info = StdioServerInfo(command="uvx", args=["mcp-server-time", "--local-timezone=Europe/Berlin"])

    mcp_toolset = None
    try:
        # Create the toolset - this will automatically discover all available tools
        mcp_toolset = MCPToolset(server_info)
        # # Check if OpenAI API key is set
        # api_key = os.environ.get("OPENAI_API_KEY")
        # if not api_key:
        #     print("OPENAI_API_KEY environment variable is not set.")
        #     print("You need to set it to run the pipeline example.")
        #     print("For now, demonstrating direct tool usage:")

        pipeline = Pipeline()
        pipeline.add_component("llm", OllamaChatGenerator(model="llama3.2", tools=mcp_toolset, agent_name="tool_llm"))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
        pipeline.add_component(
            "adapter",
            OutputAdapter(
                template="{{ initial_msg + initial_tool_messages + tool_messages }}",
                output_type=list[ChatMessage],
                unsafe=True,
            ),
        )
        pipeline.add_component("response_llm", OllamaChatGenerator(model="llama3.2", agent_name="response_llm"))
        pipeline.connect("llm.replies", "tool_invoker.messages")
        pipeline.connect("llm.replies", "adapter.initial_tool_messages")
        pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
        pipeline.connect("adapter.output", "response_llm.messages")

        user_input = "What is the time in New York? Be brief."  # can be any city
        user_input_msg = ChatMessage.from_user(text=user_input)

        result = pipeline.run({"llm": {"messages": [user_input_msg]}, "adapter": {"initial_msg": [user_input_msg]}})

        print(result["response_llm"]["replies"][0].text)

    finally:
        if mcp_toolset:
            mcp_toolset.close()


if __name__ == "__main__":
    main()
