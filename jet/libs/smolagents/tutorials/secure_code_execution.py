from typing import Optional
from smolagents import (
    CodeAgent,
    WebSearchTool,
    InferenceClientModel,
    OpenAIModel,
)

# ───────────────────────────────────────────────
#            Local model factory (as given)
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
#                   DEMO FUNCTIONS
# ───────────────────────────────────────────────

def demo_01_local_code_agent_default_executor():
    """Demo 1: Basic CodeAgent using local model + default LocalPythonExecutor (with restricted imports)"""
    model = create_local_model(temperature=0.68, max_tokens=3072)

    # Default executor → only very safe imports are allowed
    agent = CodeAgent(
        model=model,
        tools=[],
        name="safe-math-agent",
        description="Runs in restricted local python sandbox"
    )

    result = agent.run("Compute the 40th Fibonacci number using memoization.")
    print("Result:", result)


def demo_02_local_with_custom_authorized_imports():
    """Demo 2: Local model + custom allowed imports (numpy, pandas)"""
    model = create_local_model(temperature=0.7, max_tokens=4096)

    # You can pass additional_authorized_imports to CodeAgent
    agent = CodeAgent(
        model=model,
        tools=[],
        additional_authorized_imports=[
            "numpy",
            "pandas",
            "numpy.random",     # explicit submodule
            "datetime",
            "collections.*",    # wildcard example
        ],
        name="data-agent"
    )

    task = """
Create a small pandas DataFrame with columns ['date', 'value']
Add 8 rows of dummy data from 2025-01-01 onwards
Compute the rolling 3-day mean of 'value'
Print the result
"""
    result = agent.run(task)
    print("Execution result:", result)


def demo_03_local_with_blaxel_executor():
    """Demo 3: Local model + Blaxel remote sandbox for code execution"""
    model = create_local_model(temperature=0.65)

    # executor_type="blaxel" → code runs in Blaxel sandbox
    with CodeAgent(
        model=model,
        tools=[WebSearchTool()],
        executor_type="blaxel",
        name="blaxel-secure-agent"
    ) as agent:
        result = agent.run(
            "Search the web for the population of Quezon City in 2025 or most recent estimate, "
            "then compute how many people that would be per square meter if the area is ~166 km²."
        )
        print("Result:", result)


def demo_04_local_model_e2b_sandbox():
    """Demo 4: Local model + E2B sandbox for code execution (quick setup)"""
    model = create_local_model(temperature=0.72, max_tokens=4096)

    # executor_type="e2b" → code runs in E2B cloud sandbox
    with CodeAgent(
        model=model,
        tools=[],
        executor_type="e2b",
        name="e2b-protected-agent"
    ) as agent:
        result = agent.run("Calculate 2**1000 and return only the number of digits in the result.")
        print("Number of digits:", result)


def demo_05_inference_client_with_modal_executor():
    """Demo 5: Cloud model + Modal sandbox (no local factory used here)"""
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.7,
        max_tokens=4096
    )

    with CodeAgent(
        model=model,
        tools=[],
        executor_type="modal",
        name="modal-secure-compute"
    ) as agent:
        result = agent.run("What is the 35th number in the Tribonacci sequence (T_n = T_{n-1} + T_{n-2} + T_{n-3})?")
        print("Result:", result)


def demo_06_local_model_docker_executor_simple():
    """Demo 6: Local model + Docker executor (quick version)"""
    model = create_local_model(temperature=0.7)

    # Requires Docker running locally
    with CodeAgent(
        model=model,
        tools=[WebSearchTool()],
        executor_type="docker",
        name="docker-isolated-agent"
    ) as agent:
        result = agent.run("What is the current number one programming language according to recent rankings?")
        print("Result:", result)


def demo_07_wasm_executor_local_model():
    """Demo 7: Local model + WebAssembly (Pyodide) executor – browser-like isolation"""
    model = create_local_model(temperature=0.68)

    # executor_type="wasm" – very strong isolation, no OS access
    agent = CodeAgent(
        model=model,
        tools=[],
        executor_type="wasm",
        name="wasm-sandboxed-agent"
    )

    result = agent.run("Compute sum of squares of first 100 natural numbers using list comprehension.")
    print("Result:", result)


# For multi-agent examples with full sandboxing (E2B / Docker / Modal),
# see the more advanced patterns in the documentation using custom sandbox managers.

# ───────────────────────────────────────────────
#                      Usage examples
# ───────────────────────────────────────────────

if __name__ == "__main__":
    # Pick one to run (most require corresponding service accounts / docker / deno installed)
    # demo_01_local_code_agent_default_executor()
    # demo_02_local_with_custom_authorized_imports()
    # demo_03_local_with_blaxel_executor()
    # demo_04_local_model_e2b_sandbox()
    # demo_05_inference_client_with_modal_executor()
    demo_07_wasm_executor_local_model()   # usually easiest to try locally if Deno is installed
    # demo_06_local_model_docker_executor_simple()