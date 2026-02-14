# cursor_composer_equivalent_multi_agent.py

# Complete working code using smolagents to replicate Cursor IDE Composer agents features.
# This sets up multiple collaborative CodeAgents: Planner (plans next moves), Thinker (reasons/thinks), Explorer (explores web/files).
# They work together on a coding task, mimicking Cursor's multi-agent parallel execution (here simulated sequentially for simplicity).
# Tools included: Web search via DuckDuckGoSearchTool, File search/read via custom tools.
# Requires: pip install 'smolagents[toolkit]' and 'duckduckgo-search' if not included.
# For models, uses free Hugging Face Inference API (may require HF token for heavy use).

import os

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, Tool


# Custom Tool for listing files in a directory (for exploring codebase)
class FileListTool(Tool):
    name = "list_files"
    description = "List files in a given directory path."

    def forward(self, dir_path: str) -> list[str]:
        if not os.path.isdir(dir_path):
            return ["Directory not found."]
        return os.listdir(dir_path)


# Custom Tool for reading a file (for file search)
class FileReadTool(Tool):
    name = "read_file"
    description = "Read the content of a file at the given path."

    def forward(self, file_path: str) -> str:
        if not os.path.isfile(file_path):
            return "File not found."
        with open(file_path) as f:
            return f.read()


# Initialize the model (using Hugging Face Inference API, e.g., Mistral or similar)
model = InferenceClientModel(
    model="mistralai/Mistral-7B-Instruct-v0.2"
)  # Free tier model, adjust as needed

# Shared tools for all agents
tools = [DuckDuckGoSearchTool(), FileListTool(), FileReadTool()]

# Planner Agent: Plans next moves for the task
planner_agent = CodeAgent(
    tools=tools,
    model=model,
    system_prompt="You are a Planner Agent. Your role is to plan the next moves for a coding task. Break it down into steps like research, design, implementation.",
)

# Thinker Agent: Thinks/reasons about the plan or information
thinker_agent = CodeAgent(
    tools=tools,
    model=model,
    system_prompt="You are a Thinker Agent. Your role is to reason deeply about plans, information, or code. Analyze, critique, and suggest improvements.",
)

# Explorer Agent: Explores web and files to gather information
explorer_agent = CodeAgent(
    tools=tools,
    model=model,
    system_prompt="You are an Explorer Agent. Your role is to explore resources: search the web, list and read files in the codebase to gather relevant information.",
)


# Function to simulate collaborative agents working together on a task
def collaborative_coding_task(task: str, codebase_dir: str = "."):
    print(f"Starting collaborative task: {task}\n")

    # Step 1: Planner plans next moves
    print("Planner Agent is planning...")
    plan = planner_agent.run(
        f"Plan the next moves for this coding task: '{task}'. Consider exploring the codebase in '{codebase_dir}' and web if needed."
    )
    print(f"Plan:\n{plan}\n")

    # Step 2: Explorer explores based on the plan (web/file search)
    print("Explorer Agent is exploring...")
    exploration_results = explorer_agent.run(
        f"Based on this plan: '{plan}', explore the web and files in '{codebase_dir}' to gather information for '{task}'."
    )
    print(f"Exploration Results:\n{exploration_results}\n")

    # Step 3: Thinker thinks about the gathered info and plan
    print("Thinker Agent is thinking...")
    reasoning = thinker_agent.run(
        f"Reason about this plan: '{plan}' and exploration results: '{exploration_results}' for task '{task}'. Suggest refined steps or code outlines."
    )
    print(f"Reasoning:\n{reasoning}\n")

    # Optional: Could add a Builder Agent to generate code, but keeping to the three steps mentioned.
    return reasoning


# Example usage
if __name__ == "__main__":
    # Example task to replicate Cursor's coding assistance
    task = "Build a simple Python script that fetches weather data from the web and saves it to a file."
    # Assume current directory as codebase, or specify a path
    final_output = collaborative_coding_task(task)
    print(f"Final Output:\n{final_output}")
