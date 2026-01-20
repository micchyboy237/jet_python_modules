from smolagents import OpenAIModel, CodeAgent  # or ToolCallingAgent, etc.

# Connect to your local llama.cpp server
model = OpenAIModel(
    model_id="qwen3-4b-instruct-2507",           # Arbitrary but descriptive name
    api_base="http://shawn-pc.local:8080/v1",   # Your server endpoint
    api_key="sk-no-key-required",               # Dummy value (local servers usually don't check)
    temperature=0.7,
    max_tokens=2048,
    # Optional: extra kwargs forwarded to the completions call
)

# Example: Create a simple code agent
agent = CodeAgent(
    model=model,
    tools=[],  # Add your @tool decorated functions here
)

# Run the agent
agent.run("What is the capital of Japan?")