import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import Client as OllamaClient
from dotenv import load_dotenv

from deepeval.test_case import (
    MCPServer,
    MCPToolCall,
    ConversationalTestCase,
    Turn,
)

from jet.logger import logger

load_dotenv()

mcp_servers = []
turns = []


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.ollama = OllamaClient()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.input = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.input)
        )

        await self.session.initialize()

        tool_list = await self.session.list_tools()
        mcp_servers.append(
            MCPServer(
                server_name=server_script_path,
                available_tools=tool_list.tools,
            )
        )

    async def process_query(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        turns.append(Turn(role="user", content=query))

        response_text = []

        tool_response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tool_response.tools
        ]

        while True:
            response_stream = self.ollama.chat(
                model="llama3.2",
                messages=messages,
                tools=available_tools,
                options={"max_tokens": 1000},
                stream=True
            )

            tool_uses = []
            full_response_content = []

            # Ollama returns a single message with content and optional tool calls
            content = ""
            for chunk in response_stream:
                logger.teal(chunk['message']['content'], flush=True)
                content += chunk['message']['content']

            full_response_content.append(content)

            if content:
                response_text.append(content)
                turns.append(Turn(role="assistant", content=content))

            if chunk.get("tool_calls"):
                tool_uses.extend(chunk["tool_calls"])

            messages.append(
                {"role": "assistant", "content": content, "tool_calls": chunk.get("tool_calls", [])}
            )

            if not tool_uses:
                break

            for tool_use in tool_uses:
                tool_name = tool_use["function"]["name"]
                tool_args = tool_use["function"]["arguments"]
                tool_id = tool_use.get("id", f"tool_{len(messages)}")

                result = await self.session.call_tool(tool_name, tool_args)
                tool_called = MCPToolCall(
                    name=tool_name, args=tool_args, result=result
                )

                turns.append(
                    Turn(
                        role="assistant",
                        content=f"Tool call: {tool_name} with args {tool_args}",
                        mcp_tools_called=[tool_called],
                    )
                )

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result.content,
                            }
                        ],
                    }
                )

        return "\n".join(response_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            query = input("Query: ")

            if query.lower() == "quit":
                convo_test_case = ConversationalTestCase(
                    turns=turns, mcp_servers=mcp_servers
                )
                print(convo_test_case)
                print("-" * 50)
                break

            response = await self.process_query(query)
            print("\n" + response)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    DEFAULT_SERVER_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/servers/mcp/mcp_server.py"
    server_path = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_SERVER_PATH
    if len(sys.argv) < 2:
        print(f"No server script provided. Using default: {DEFAULT_SERVER_PATH}")

    client = MCPClient()
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
