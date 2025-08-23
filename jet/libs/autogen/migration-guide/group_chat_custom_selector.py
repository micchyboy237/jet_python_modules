"""Implement a group chat with a custom selector in AutoGen v0.4.

This module demonstrates a `SelectorGroupChat` in AutoGen v0.4 with a custom `selector_func` to control agent interactions, replacing the v0.2 `GroupChatManager` with custom speaker selection. It uses planning, web search, and data analyst agents to collaboratively solve a task with tool usage.
"""

import os
import shutil
import asyncio
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from jet.file.utils import save_file
from jet.llm.mlx.adapters.mlx_autogen_chat_llm_adapter import MLXAutogenChatLLMAdapter
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchAPIWrapper, DuckDuckGoSearchRun

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def search_web_tool(query: str) -> str:
    # Initialize the DuckDuckGoSearchAPIWrapper with custom parameters
    api_wrapper = DuckDuckGoSearchAPIWrapper(
        region="wt-wt",  # Worldwide region
        safesearch="moderate",  # Moderate safe search
        time="y",  # Results from the past year
        max_results=5,  # Maximum of 5 results
        source="text"  # Text search
    )
    # Initialize the DuckDuckGoSearchRun tool
    search_tool = DuckDuckGoSearchRun(api_wrapper=api_wrapper)
    result = search_tool._run(query)
    return result


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


def create_team(model_client: MLXAutogenChatLLMAdapter) -> SelectorGroupChat:
    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            Web search agent: Searches for information
            Data analyst: Performs calculations
        You only plan and delegate tasks - you do not execute them yourself.
        When assigning tasks, use this format:
        1. <agent> : <task>
        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )
    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="A web search agent.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )
    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="A data analyst agent. Useful for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        """,
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
        if messages[-1].source != planning_agent.name:
            return planning_agent.name
        return None
    team = SelectorGroupChat(
        [planning_agent, web_search_agent, data_analyst_agent],
        model_client=MLXAutogenChatLLMAdapter(
            model="llama-3.2-3b-instruct-4bit", log_dir=f"{OUTPUT_DIR}/group_chats"),
        termination_condition=termination,
        selector_func=selector_func,
    )
    return team


async def main() -> None:
    model_client = MLXAutogenChatLLMAdapter(
        model="llama-3.2-1b-instruct-4bit", log_dir=f"{OUTPUT_DIR}/chats")
    team = create_team(model_client)
    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    await Console(team.run_stream(task=task))
    state = await team.save_state()
    save_file(state, f"{OUTPUT_DIR}/group_chat_state.json")

asyncio.run(main())
