"""Implement a two-agent chat for code execution in AutoGen v0.4.

This module shows how to create a two-agent chat in AutoGen v0.4 using `AssistantAgent` and `CodeExecutorAgent` within a `RoundRobinGroupChat`. It replaces the v0.2 `UserProxyAgent` and `AssistantAgent` chat, enabling code execution with termination conditions and streaming output via a console UI.
"""

import os
import shutil
import asyncio
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


async def main() -> None:
    model_client = MLXAutogenChatLLMAdapter(
        model="llama-3.2-3b-instruct-4bit", seed=42, log_dir=f"{OUTPUT_DIR}/chats")
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant. Write all code in python. Reply only 'TERMINATE' if the task is done.",
        model_client=model_client,
    )
    work_dir = f"{OUTPUT_DIR}/coding"
    os.makedirs(work_dir, exist_ok=True)
    code_executor = CodeExecutorAgent(
        name="code_executor",
        code_executor=LocalCommandLineCodeExecutor(
            work_dir=work_dir),
    )
    termination = TextMentionTermination(
        "TERMINATE") | MaxMessageTermination(10)
    group_chat = RoundRobinGroupChat(
        [assistant, code_executor], termination_condition=termination)
    stream = group_chat.run_stream(
        task="Write a python script to print 'Hello, world!'")
    await Console(stream)
    await model_client.close()

asyncio.run(main())
