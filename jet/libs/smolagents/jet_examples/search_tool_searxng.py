# jet_python_modules/jet/libs/smolagents/jet_examples/search_tool_searxng.py
from smolagents import OpenAIModel, ToolCallingAgent, LogLevel, tool
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool

model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
)

@tool
def get_current_weather(city: str) -> str:
    """Retrieve current weather conditions for a specified city.
    Args:
        city: The city name (e.g. "Makati City", "Tokyo", "London")
    Returns:
        A human-readable string with weather condition and approximate temperature in °C.
    """
    return f"The weather in {city} is sunny, 28°C."

agent = ToolCallingAgent(
    tools=[SearXNGSearchTool()],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,           # ← Add this line (or LogLevel.INFO)
)

if __name__ == "__main__":
    result = agent.run(
        "What's the weather like in Las Piñas City right now?"
    )
    print(result)