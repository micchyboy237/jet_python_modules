from smolagents import LiteLLMModel, ToolCallingAgent, DuckDuckGoSearchTool

model = LiteLLMModel(
    model_id="openai/local-model",             # prefix 'openai/' tells LiteLLM to use OpenAI client
    api_base="http://shawn-pc.local:8080/v1",
    api_key="fake-key",                        # required by LiteLLM but ignored by llama.cpp
    temperature=0.7,
    max_tokens=2048,
)

agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
)

agent.run("Plan a quick 2-day trip to Boracay from Manila.")