from swarms.agents.i_agent import IterativeReflectiveExpansion

agent = IterativeReflectiveExpansion(
    max_iterations=1,
    model_name="ollama/llama3.2",
)

agent.run("What is the 40th prime number?")
