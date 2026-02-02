# main.py
from agent_factory import create_memory_enabled_agent

if __name__ == "__main__":
    agent = create_memory_enabled_agent(verbosity=1)

    # Example 1 – basic math + manual save
    agent.run("""
    Compute the 15th Fibonacci number.
    Then save the result as an important fact named "fib_15_result".
    """)

    # Example 2 – recall previous knowledge
    agent.run("""
    What was the 15th Fibonacci number we computed earlier?
    Use long-term memory if needed.
    """)

    # Example 3 – shared state usage
    agent.run("""
    The current project name is "Aether". Save it to shared state under key "project_name".
    Then read back the project name from shared state.
    """)

    # Example 4 – multi-turn awareness via shared state
    agent.run("What is the current project name?")
