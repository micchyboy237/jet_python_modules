"""
Simple example of using MCP client with browser-use.

This example shows how to connect to an MCP server and use its tools with an agent.
"""

import asyncio
import os

from browser_use import Agent, Controller
from browser_use.llm.ollama.chat import ChatOllama
from browser_use.mcp.client import MCPClient


async def main():
	# Initialize controller
	controller = Controller()

	# Connect to a filesystem MCP server
	# This server provides tools to read/write files in a directory
	server_base_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/browser-use/examples"
	mcp_client = MCPClient(
		server_name='filesystem', command='npx', args=['@modelcontextprotocol/server-filesystem', server_base_dir]
	)

	# Connect and register MCP tools
	await mcp_client.connect()
	await mcp_client.reg(controller)

	# Create agent with MCP-enabled controller
	agent = Agent(
		task='List all files on the Desktop and read the content of any .txt files you find',
		llm=ChatOllama(model='llama3.2'),
		controller=controller,
	)

	# Run the agent - it now has access to filesystem tools
	await agent.run()

	# Disconnect when done
	await mcp_client.disconnect()


if __name__ == '__main__':
	asyncio.run(main())
