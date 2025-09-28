from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
# from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo
# from ipywidgets import Video
from jet.adapters.haystack.ollama_chat_generator import OllamaChatGenerator
from jet.file.utils import save_file
from jet.logger import logger
import os
import shutil

from jet.transformers.formatters import format_json


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

SERVER_INFO = StreamableHttpServerInfo(url="http://localhost:8931/mcp")

SYSTEM_MESSAGE = """
You are an intelligent assistant equipped with tools for navigating the web.

You can use tools when appropriate, but not every task requires them — you also have strong reasoning and language capabilities.
If a request seems challenging, don't default to refusal due to perceived tool limitations. Instead, think creatively and attempt a solution using the skills you do have.
You are more capable than you might assume. Trust your abilities.
"""

def create_agent(agent_name: str, tool_names: list[str]) -> Agent:
    toolset = MCPToolset(server_info=SERVER_INFO, tool_names=tool_names)
    logger.debug("Toolset:")
    logger.debug(format_json(toolset.tools))
    chat_generator = OllamaChatGenerator(model="llama3.2", agent_name=agent_name, verbose=False, generation_kwargs={"temperature": 0.1})

    agent = Agent(
        chat_generator=chat_generator,
        tools=toolset,
        system_prompt=SYSTEM_MESSAGE,
        exit_conditions=["text"],
        streaming_callback=print_streaming_chunk
    )

    return agent

def example():
    tool_names = ["browser_navigate", "browser_snapshot"]

    messages = [ChatMessage.from_user("""
    1. Navigate to https://www.nbcnews.com and capture an accessibility snapshot of the page.
    2. Given the snapshot, write doen all relevant links.
    """)]
    agent = create_agent("MCP_Agent", tool_names=tool_names)
    result = agent.run(messages=messages)
    save_file(result["last_message"].text, f"{OUTPUT_DIR}/results/last_message.md")

if __name__ == "__main__":
    example()

    logger.info("\n\n[DONE]", bright=True)
