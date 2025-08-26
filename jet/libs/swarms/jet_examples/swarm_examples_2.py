from swarms import Agent
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
import ollama
import os
import json
import yaml
import xml.etree.ElementTree as ET


# Example 8: Agent with Template and Retry Mechanism
def template_retry_example() -> str:
    """Demonstrates template, retry_attempts, retry_interval, and return_history."""
    template = "Task: {task}\nProvide a concise response."
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        template=template,
        retry_attempts=2,
        retry_interval=1,
        return_history=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Summarize market trends.")
    return response


# Example 9: Agent with Stopping Token and Dynamic Loops
def stopping_token_dynamic_loops_example() -> str:
    """Demonstrates stopping_token, dynamic_loops, and loop_interval."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        stopping_token="<STOP>",
        dynamic_loops=True,
        loop_interval=2,
        max_loops=3,
        verbose=True
    )
    response = agent.run("Generate a report ending with <STOP>.")
    return response


# Example 10: Agent with Dashboard and Custom Exit Command
def dashboard_exit_command_example() -> str:
    """Demonstrates dashboard, agent_name, agent_description, and custom_exit_command."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        dashboard=True,
        agent_name="FinanceBot",
        agent_description="A bot for financial analysis.",
        custom_exit_command="exit_now",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Show financial dashboard.")
    return response


# Example 11: Agent with SOP and Autosave
def sop_autosave_example() -> str:
    """Demonstrates sop, sop_list, autosave, and saved_state_path."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        sop="Follow financial analysis SOP.",
        sop_list=["Step 1: Analyze data", "Step 2: Generate report"],
        autosave=True,
        saved_state_path="./agent_state.json",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Execute SOP for financial analysis.")
    agent.save_state()
    return response


# Example 12: Agent with Self-Healing and Code Interpreter
def self_healing_code_interpreter_example() -> str:
    """Demonstrates self_healing_enabled, code_interpreter, and run_with_timeout."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        self_healing_enabled=True,
        code_interpreter=True,
        timeout=30,
        max_loops=1,
        verbose=True
    )
    response = agent.run_with_timeout(
        "Execute Python code: print('Financial data')")
    return response


# Example 13: Agent with Multi-Modal and PDF Ingestion
def multi_modal_pdf_example() -> str:
    """Demonstrates multi_modal, pdf_path, list_of_pdf, and tokenizer."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        multi_modal=True,
        pdf_path="financial_report.pdf",
        list_of_pdf=["financial_report.pdf"],
        tokenizer=None,  # Placeholder, assumes tokenizer object
        max_loops=1,
        verbose=True
    )
    response = agent.run("Summarize the PDF content.")
    return response


# Example 14: Agent with Callbacks and Metadata
def callbacks_metadata_example() -> str:
    """Demonstrates callback, callbacks, metadata, and metadata_output_type."""
    def sample_callback(response: str) -> None:
        print(f"Callback received: {response}")

    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        callback=sample_callback,
        callbacks=[sample_callback],
        metadata={"task_id": "123"},
        metadata_output_type="json",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate metadata for task.")
    return response


# Example 15: Agent with Search Algorithm and Evaluator
def search_evaluator_example() -> str:
    """Demonstrates search_algorithm, evaluator, and best_of_n."""
    def sample_search_algorithm(query: str) -> List[str]:
        return [f"Result for {query}"]

    def sample_evaluator(response: str) -> float:
        return 0.9 if "result" in response.lower() else 0.1

    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        search_algorithm=sample_search_algorithm,
        evaluator=sample_evaluator,
        best_of_n=2,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Search for market trends.")
    return response


# Example 16: Agent with Logging and Custom Loop Condition
def logging_custom_loop_example() -> str:
    """Demonstrates logs_to_filename, log_directory, and custom_loop_condition."""
    def custom_loop_condition(response: str) -> bool:
        return "continue" not in response.lower()

    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        logs_to_filename="agent_log.txt",
        log_directory="./logs",
        custom_loop_condition=custom_loop_condition,
        max_loops=2,
        verbose=True
    )
    response = agent.run("Generate report, exclude 'continue'.")
    return response


# Example 17: Agent with Function Calling and Output Cleaner
def function_calling_cleaner_example() -> str:
    """Demonstrates function_calling_type, function_calling_format_type, and output_cleaner."""
    def output_cleaner(response: str) -> str:
        return response.strip().upper()

    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        function_calling_type="json",
        function_calling_format_type="json",
        output_cleaner=output_cleaner,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a JSON response.")
    return response


# Example 18: Agent with Planning and Custom Tools Prompt
def planning_tools_prompt_example() -> str:
    """Demonstrates planning, planning_prompt, custom_tools_prompt, and tool_system_prompt."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        planning="Plan financial analysis steps.",
        planning_prompt="Create a step-by-step plan for the task.",
        custom_tools_prompt=lambda x: f"Tool prompt: {x}",
        tool_system_prompt="Use tools for financial analysis.",
        max_loops=1,
        verbose=True
    )
    response = agent.run("Plan a financial analysis.")
    return response


# Example 19: Agent with Advanced Parameters
def advanced_parameters_example() -> str:
    """Demonstrates frequency_penalty, presence_penalty, temperature, and max_tokens."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        frequency_penalty=0.5,
        presence_penalty=0.3,
        temperature=0.7,
        max_tokens=100,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a concise financial summary.")
    return response


# Example 20: Agent with Scheduled Run and Workspace
def scheduled_workspace_example() -> str:
    """Demonstrates scheduled_run_date, workspace_dir, and agent_ops_on."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        scheduled_run_date=datetime(2025, 8, 27, 3, 37),
        workspace_dir="./workspace",
        agent_ops_on=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Run scheduled task.")
    return response

# Example 21: Agent with Bulk Run and Concurrent Execution


def bulk_concurrent_example() -> List[str]:
    """Demonstrates bulk_run, run_concurrent, and run_async."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        max_loops=1,
        verbose=True
    )
    tasks = ["Task 1: Summary", "Task 2: Analysis"]
    response = agent.bulk_run(tasks)
    return response


# Example 22: Agent with History and Feedback Analysis
def history_feedback_example() -> str:
    """Demonstrates print_history_and_memory, analyze_feedback, and truncate_history."""
    agent = Agent(
        llm=ollama,
        model_name="llama3.2",
        return_history=True,
        max_loops=1,
        verbose=True
    )
    response = agent.run("Generate a report.")
    agent.print_history_and_memory()
    agent.analyze_feedback("Feedback: Good report")
    agent.truncate_history()
    return response
