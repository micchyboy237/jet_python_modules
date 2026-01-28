from typing import Optional
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    WebSearchTool,
    VisitWebpageTool,
    InferenceClientModel,
    OpenAIModel,           # assuming this is the compatible client for local openai-like servers
)

# ───────────────────────────────────────────────
#            Factory (as given)
# ───────────────────────────────────────────────
def create_local_model(
    temperature: float = 0.7,
    max_tokens: Optional[int] = 2048,
    model_id: str = "local-model",
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        base_url="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ───────────────────────────────────────────────
#             DEMO FUNCTIONS
# ───────────────────────────────────────────────

def demo_01_local_code_agent_basic():
    """Demo 1: Simple CodeAgent using local llama.cpp server"""
    model = create_local_model(temperature=0.75, max_tokens=3072)
    
    agent = CodeAgent(
        model=model,
        tools=[],
        name="local-math-agent",
        description="Local code interpreter style agent"
    )
    
    result = agent.run("What is 19**7 modulo 10**9+7?")
    print("Result:", result)


def demo_02_local_toolcalling_agent():
    """Demo 2: Tool-calling agent with web tools on local model"""
    model = create_local_model(temperature=0.6, max_tokens=4096)
    
    agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="local-search-agent",
        description="Local model with web browsing capability"
    )
    
    result = agent.run("What was the closing price of TSLA on its most recent trading day?")
    print("Result:", result)


def demo_03_local_manager_with_subagent():
    """Demo 3: Manager (CodeAgent) → managed ToolCallingAgent (both local)"""
    model = create_local_model(temperature=0.7, max_tokens=4096)
    
    search_agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="web_helper",
        description="Performs web searches and page reading"
    )
    
    manager = CodeAgent(
        model=model,
        managed_agents=[search_agent],
        tools=[],
        name="coordinating_agent"
    )
    
    result = manager.run(
        "If the US keeps its 2024 real GDP growth rate, how many years to double?"
    )
    print("Doubling time estimate:", result)


def demo_04_hf_inference_client_cloud():
    """Demo 4: Using InferenceClientModel (Hugging Face hosted inference)"""
    # No local factory used here — different backend
    model = InferenceClientModel(
        # model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",   # example
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature=0.7,
        max_tokens=4096
    )
    
    agent = ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],
        model=model,
        name="cloud-search-agent"
    )
    
    result = agent.run("Latest stable diffusion version as of today?")
    print(result)


def demo_05_local_phoenix_tracing():
    """Demo 5: Local model + Phoenix tracing"""
    from phoenix.otel import register
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    
    # Start tracing
    register()
    SmolagentsInstrumentor().instrument()
    
    model = create_local_model(temperature=0.68)
    
    agent = CodeAgent(model=model, managed_agents=[
        ToolCallingAgent(
            tools=[WebSearchTool()],
            model=model,
            name="fast-lookup"
        )
    ])
    
    agent.run("Current president of Brazil?")


def demo_06_langfuse_tracing_local():
    """Demo 6: Local model + Langfuse (minimal version)"""
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    
    # Make sure these are set in environment or here
    # os.environ["LANGFUSE_PUBLIC_KEY"]  = "..."
    # os.environ["LANGFUSE_SECRET_KEY"]  = "..."
    # os.environ["LANGFUSE_HOST"]        = "https://cloud.langfuse.com"
    
    SmolagentsInstrumentor().instrument()
    
    model = create_local_model(temperature=0.7)
    
    agent = ToolCallingAgent(
        model=model,
        tools=[WebSearchTool(), VisitWebpageTool()],
        name="traced-local-agent"
    )
    
    agent.run("What is the current market cap rank of SOL (Solana)?")


# ───────────────────────────────────────────────
#                 Usage examples
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # Pick one or run them sequentially
    # demo_01_local_code_agent_basic()
    # demo_02_local_toolcalling_agent()
    demo_03_local_manager_with_subagent()
    # demo_04_hf_inference_client_cloud()
    # demo_05_local_phoenix_tracing()
    # demo_06_langfuse_tracing_local()