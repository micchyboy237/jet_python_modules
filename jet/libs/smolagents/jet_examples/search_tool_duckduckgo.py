from smolagents import OpenAIModel, ToolCallingAgent, tool, DuckDuckGoSearchTool


# 1. Create the model pointing to your local llama.cpp server
model = OpenAIModel(
    model_id="local-model",
    api_base="http://shawn-pc.local:8080/v1",
    api_key="not-needed",
    temperature=0.7,
    max_tokens=2048,
)


# 2. Define custom tool with proper docstring format
@tool
def get_current_weather(city: str) -> str:
    """Retrieve current weather conditions for a specified city.

    Args:
        city: The city name (e.g. "Makati City", "Tokyo", "London")

    Returns:
        A human-readable string with weather condition and approximate temperature in °C.
    """
    # In real code → call weather API
    return f"The weather in {city} is sunny, 28°C."


# 3. Create the agent
agent = ToolCallingAgent(
    tools=[get_current_weather, DuckDuckGoSearchTool()],
    model=model,
    add_base_tools=False,
)


# 4. Run example query
if __name__ == "__main__":
    result = agent.run(
        "What's the weather like in Makati City right now?"
    )
    print(result)