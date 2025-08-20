import asyncio
import os
from jet.logger import CustomLogger
from jet.servers.mcp.llm_client import query_llm
from jet.servers.mcp.tool_executor import discover_tools, process_tool_requests

LOGS_DIR = os.path.expanduser("~/playwright_mcp/logs")
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)


async def chat_session():
    tools = await discover_tools()
    logger.debug(
        f"Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            logger.debug("Ending chat session.")
            break
        llm_response, messages, tool_requests = await query_llm(user_input, tools, messages, logger)
        if tool_requests:
            tool_outputs = await process_tool_requests(tool_requests, tools, messages, logger)
            follow_up_messages = [
                {"role": "system", "content": llm_response}] + messages
            llm_response, messages, _ = await query_llm("", tools, follow_up_messages, logger)
        print(f"Assistant: {llm_response}")

if __name__ == "__main__":
    asyncio.run(chat_session())
