"""
Demo: Autonomous Agent with Tool Use & OpenAI Streaming
Covers: @agent, @tool, @chain, nested spans, streaming LLM planning
"""

import asyncio
import json

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL, PHOENIX_BASE_URL
from jet_telemetry import agent, initialize_telemetry, llm, tool
from openai import AsyncOpenAI

initialize_telemetry(service_name="agent-demo", endpoint=PHOENIX_BASE_URL)

llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")


@tool(description="Get current weather for a city")
async def get_weather(city: str) -> dict:
    await asyncio.sleep(0.1)
    return {"city": city, "temp_c": 22, "condition": "partly cloudy"}


@tool(description="Search knowledge base for technical docs")
async def search_docs(query: str) -> list[str]:
    await asyncio.sleep(0.08)
    return [
        f"Doc snippet about '{query}' - section 3.2",
        f"Reference manual entry for '{query}'",
    ]


@llm(model_name=LLM_MODEL)
async def plan_next_step(task: str, history: list[dict]) -> dict:
    """Streaming LLM call for agent planning."""
    tools_schema = [
        {"name": "get_weather", "description": "Get current weather for a city"},
        {
            "name": "search_docs",
            "description": "Search knowledge base for technical docs",
        },
    ]
    messages = [
        {
            "role": "system",
            "content": 'You are a planning agent. Respond with JSON: {"action": "tool_name|done", "args": {...}, "reasoning": "..."}',
        },
        {
            "role": "user",
            "content": f"Task: {task}\nHistory: {json.dumps(history[-3:])}\nAvailable tools: {json.dumps(tools_schema)}",
        },
    ]

    print("\n🧠 Agent Thinking: ", end="", flush=True)
    collected_content = []

    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                continue
            if delta.content:
                collected_content.append(delta.content)
                print(delta.content, end="", flush=True)

    print("\n", flush=True)
    full_response = "".join(collected_content)

    # Extract JSON from potentially verbose response
    try:
        start = full_response.find("{")
        end = full_response.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(full_response[start:end])
        return json.loads(full_response)
    except json.JSONDecodeError:
        return {
            "action": "done",
            "reasoning": f"Failed to parse JSON: {full_response[:100]}",
        }


@agent(name="research-agent")
async def research_agent(task: str, max_steps: int = 5) -> str:
    """Autonomous agent that plans, uses tools, and synthesizes answers."""
    history = []

    for step in range(max_steps):
        plan = await plan_next_step(task, history)
        history.append({"step": step, "plan": plan})

        if plan.get("action") == "done":
            return plan.get("reasoning", "Task completed.")

        tool_map = {"get_weather": get_weather, "search_docs": search_docs}
        selected_tool = tool_map.get(plan.get("action"))

        if selected_tool:
            try:
                result = await selected_tool(**plan.get("args", {}))
                history.append({"step": step, "tool_result": result})
            except Exception as e:
                history.append({"step": step, "error": str(e)})

    return "Max steps reached. Partial results in history."


async def main():
    result = await research_agent(
        "What's the weather in Tokyo and how does it relate to server cooling efficiency?"
    )
    print(f"\n🤖 Agent Result: {result}")
    print(f"🔍 View agent trace at: {PHOENIX_BASE_URL}")


if __name__ == "__main__":
    asyncio.run(main())
