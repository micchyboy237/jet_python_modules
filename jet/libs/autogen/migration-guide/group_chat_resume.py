"""Resume a group chat with saved state in AutoGen v0.4.

This module shows how to create a `RoundRobinGroupChat` with writer and critic agents in AutoGen v0.4, save its state, and resume it for a new task. It replaces the v0.2 manual message saving, using `save_state` and `load_state` to persist and restore the group chat’s history.
"""

import os
import shutil
import asyncio
import json
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from jet.data.utils import generate_unique_hash
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def create_team() -> RoundRobinGroupChat:
    model_client_writer = create_model_client()
    writer = AssistantAgent(
        name="writer",
        description="A writer.",
        system_message="You are a writer.",
        model_client=model_client_writer,
    )

    model_client_critic = create_model_client()
    critic = AssistantAgent(
        name="critic",
        description="A critic.",
        system_message="You are a critic, provide feedback on the writing. Reply only 'APPROVE' if the task is done.",
        model_client=model_client_critic,
    )

    termination = TextMentionTermination("APPROVE")
    group_chat = RoundRobinGroupChat(
        [writer, critic], termination_condition=termination)

    return group_chat


def create_model_client() -> MLXAutogenChatLLMAdapter:
    session_id = generate_unique_hash()
    model_client = MLXAutogenChatLLMAdapter(
        model="llama-3.2-1b-instruct-4bit", seed=42, session_id=session_id, log_dir=f"{OUTPUT_DIR}/chats")
    return model_client


async def main() -> None:
    group_chat = create_team()
    stream = group_chat.run_stream(
        task="Write a short story about a robot that discovers it has feelings.")
    await Console(stream)
    state = await group_chat.save_state()
    with open(f"{OUTPUT_DIR}/group_chat_state.json", "w") as f:
        json.dump(state, f)
    group_chat = create_team()
    with open(f"{OUTPUT_DIR}/group_chat_state.json", "r") as f:
        state = json.load(f)
    await group_chat.load_state(state)
    stream = group_chat.run_stream(task="Translate the story into Tagalog.")
    await Console(stream)

asyncio.run(main())
