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


@tool(description="Get current weather for a city. Args: city (str)")
async def get_weather(city: str) -> dict:
    """Simulated weather tool."""
    await asyncio.sleep(0.1)
    # Return a consistent structure
    return {"city": city, "temp_c": 22, "condition": "partly cloudy", "humidity": "60%"}


@tool(description="Search knowledge base for technical docs. Args: query (str)")
async def search_docs(query: str) -> list[str]:
    """Simulated search tool."""
    await asyncio.sleep(0.08)
    return [
        f"Doc snippet about '{query}' - section 3.2",
        f"Reference manual entry for '{query}'",
    ]


@llm(model_name=LLM_MODEL)
async def plan_next_step(task: str, history: list[dict]) -> dict:
    """Streaming LLM call for agent planning."""
    # Define tools clearly for the LLM
    tools_description = """
    Available Tools:
    1. get_weather(city: str): Get current weather.
    2. search_docs(query: str): Search technical documentation.
    
    Respond with ONLY valid JSON:
    {
      "action": "tool_name" or "final_answer",
      "args": {"arg_name": "value"},
      "reasoning": "Why you chose this action"
    }
    """

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant. {tools_description}",
        },
        {
            "role": "user",
            "content": f"Task: {task}\n\nHistory:\n{json.dumps(history[-2:], indent=2)}",
        },
    ]

    print("\n🧠 Agent Thinking: ", end="", flush=True)
    collected_content = []

    try:
        stream = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            stream=True,
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

    except Exception as e:
        print(f"\n❌ LLM Error: {e}", flush=True)
        return {"action": "final_answer", "reasoning": f"LLM Error: {str(e)}"}

    print("\n", flush=True)
    full_response = "".join(collected_content)

    # Robust JSON extraction
    try:
        # Try to find JSON block if wrapped in markdown or text
        start = full_response.find("{")
        end = full_response.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(full_response[start:end])
        return json.loads(full_response)
    except json.JSONDecodeError:
        return {
            "action": "final_answer",
            "reasoning": f"Could not parse JSON from: {full_response[:100]}",
        }


@agent(name="research-agent")
async def research_agent(task: str, max_steps: int = 5) -> str:
    """Autonomous agent that plans, uses tools, and synthesizes answers."""
    history = []
    tool_map = {"get_weather": get_weather, "search_docs": search_docs}

    for step in range(max_steps):
        plan = await plan_next_step(task, history)
        action = plan.get("action", "")

        # Check for completion
        if action == "final_answer" or action == "done":
            return plan.get("reasoning", "Task completed.")

        # Execute tool
        selected_tool = tool_map.get(action)
        if selected_tool:
            try:
                args = plan.get("args", {})
                # Ensure args match function signature (simple validation)
                result = await selected_tool(**args)
                history.append({"step": step, "action": action, "result": result})
                print(f"✅ Tool '{action}' executed successfully.", flush=True)
            except Exception as e:
                error_msg = f"Tool '{action}' failed: {str(e)}"
                history.append({"step": step, "action": action, "error": error_msg})
                print(f"❌ {error_msg}", flush=True)
        else:
            history.append({"step": step, "error": f"Unknown action: {action}"})
            print(f"❌ Unknown action: {action}", flush=True)

    return "Max steps reached. See history for details."


async def main():
    result = await research_agent("What's the weather in Tokyo?")
    print(f"\n🤖 Final Result: {result}")
    print(f"🔍 View agent trace at: {PHOENIX_BASE_URL}")


if __name__ == "__main__":
    asyncio.run(main())
