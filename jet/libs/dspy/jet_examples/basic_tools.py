import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()


# Define your tools as functions
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny and 75°F"


def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': [relevant information...]"


# Create a ReAct agent
react_agent = dspy.ReAct(
    signature="question -> answer", tools=[get_weather, search_web], max_iters=5
)

# Use the agent
result = react_agent(question="What's the weather like in Tokyo?")
print(result.answer)
print("Tool calls made:", result.trajectory)
