"""
Demo: Autonomous Agent with Tool Use & OpenAI Streaming
Covers: @agent, @tool, @chain, nested spans, streaming LLM planning
Span Hierarchy:
📦 research-agent (AGENT)
│
├── 🧠 plan_next_step (LLM) [Iteration 1]
│   ├── attr: llm.model_name = "llama-3.2-3b-instruct"
│   └── attr: llm.input_messages = [...]
│
├── 🛠️ get_weather (TOOL)
│   ├── attr: tool.name = "get_weather"
│   ├── attr: tool.description = "Get current weather..."
│   └── attr: tool.parameters = {"city": "Tokyo"}
│
├── 🧠 plan_next_step (LLM) [Iteration 2]
│   └── (Decides to finish or call another tool)
│
└── ... (Continues until max_steps or final_answer)
"""

import asyncio
import json

from jet.adapters.llama_cpp.config import LLM_BASE_URL, LLM_MODEL, PHOENIX_BASE_URL
from jet_telemetry import agent, get_trace_url, initialize_telemetry, llm, tool
from openai import AsyncOpenAI

initialize_telemetry(service_name="agent-demo", endpoint=PHOENIX_BASE_URL)
llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")


def log_trace_download(trace_url: str | None):
    """Helper to log trace URL and trigger JSONL export if possible."""
    if not trace_url:
        return

    print(f"🔍 View complete trace: {trace_url}")

    # Attempt to export JSONL using the helper from jet_telemetry if available
    try:
        from jet_telemetry import export_spans_to_jsonl

        # Extract trace_id from URL for filtering
        if "/redirects/traces/" in trace_url:
            trace_id = trace_url.split("/redirects/traces/")[-1]
            project_name = "agent-demo"  # Match service_name

            jsonl_path = export_spans_to_jsonl(
                project_name=project_name,
                trace_id=trace_id,
                output_path=f"traces/{trace_id}.jsonl",
                phoenix_base_url=PHOENIX_BASE_URL,
                wait_for_flush=True,
            )
            if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                print(f"📥 Exported JSONL: {jsonl_path.resolve()}")
    except Exception as e:
        print(f"⚠️ Could not auto-export JSONL: {e}")


@tool(description="Get current weather for a city. Args: city (str)")
async def get_weather(city: str) -> dict:
    """Simulated weather tool."""
    await asyncio.sleep(0.1)
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
            "content": f"Task: {task}\nHistory:\n{json.dumps(history[-2:], indent=2)}",
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
    try:
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
async def research_agent(task: str, max_steps: int = 5) -> dict:
    """Autonomous agent that plans, uses tools, and synthesizes answers."""
    history = []
    tool_map = {"get_weather": get_weather, "search_docs": search_docs}
    for step in range(max_steps):
        plan = await plan_next_step(task, history)
        action = plan.get("action", "")
        if action == "final_answer" or action == "done":
            return {
                "result": plan.get("reasoning", "Task completed."),
                "trace_url": get_trace_url(PHOENIX_BASE_URL),
            }
        selected_tool = tool_map.get(action)
        if selected_tool:
            try:
                args = plan.get("args", {})
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
    return {
        "result": "Max steps reached. See history for details.",
        "trace_url": get_trace_url(PHOENIX_BASE_URL),
    }


async def main():
    result_dict = await research_agent("What's the weather in Tokyo?")
    print(f"\n🤖 Final Result: {result_dict['result']}")
    log_trace_download(result_dict.get("trace_url"))


if __name__ == "__main__":
    asyncio.run(main())
