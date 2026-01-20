from smolagents import CodeAgent, OpenAIModel, tool, DuckDuckGoSearchTool
from rich.live import Live                  # ← nice live updating console
from rich.text import Text
from rich.console import Console

console = Console()

# Your local llama.cpp server
model = OpenAIModel(
    model_id="whatever",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="lm-studio",
    temperature=0.4,
    max_tokens=2048,
    stream=True,               # important: tell the model wrapper to stream
)

@tool
def get_current_weather(city: str) -> str:
    """Retrieve current weather conditions for a specified city.

    Args:
        city: The city name (e.g. "Makati City", "Tokyo")

    Returns:
        A human-readable string with weather condition and approximate temperature in °C.
    """
    return f"The weather in {city} is sunny, 28°C."

agent = CodeAgent(
    tools=[get_current_weather, DuckDuckGoSearchTool()],
    model=model,
    add_base_tools=False,
)

# ── Streaming with rich live console ──
with Live("", refresh_per_second=8, console=console) as live:
    full_response = ""
    
    for step in agent.run(
        "What's the weather like in Makati City right now? "
        "Also tell me something interesting about Makati.",
        stream=True
    ):
        # step is usually a dict-like object with 'action', 'thought', 'observation', etc.
        # but for final answer streaming we usually want the content delta
        if hasattr(step, 'content') and step.content:          # final answer chunks
            delta = step.content
            full_response += delta
            live.update(Text(full_response))
            
        elif hasattr(step, 'thought'):                         # show thinking steps too
            live.update(Text(f"[thinking] {step.thought}\n\n{full_response}"))

    # Final update
    live.update(Text(full_response))