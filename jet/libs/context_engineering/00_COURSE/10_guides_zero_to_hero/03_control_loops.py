#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context-Engineering: Control Loops for Multi-Step LLM Interactions
=================================================================

This module demonstrates how to implement control flow mechanisms
for orchestrating complex multi-step LLM interactions. Building on
the context expansion techniques from previous notebooks, we now
explore patterns for:

1. Sequential chaining (output of one step → input to next)
2. Iterative refinement (improving a response through cycles)
3. Conditional branching (different paths based on LLM output)
4. Self-critique and correction (meta-evaluation of outputs)
5. External validation loops (using tools/knowledge to verify)

The patterns are implemented with a focus on token efficiency and
maintaining context coherence across steps.

Usage:
    # In Jupyter or Colab:
    %run 03_control_loops.py
    # or
    from control_loops import SequentialChain, IterativeRefiner, ConditionalBrancher
"""

import os
import re
import json
import time
import tiktoken
import pathlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar

# Type variables for better type hinting
T = TypeVar('T')
Response = Union[str, Dict[str, Any]]

# For logging and visualization
import logging
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML  # Keep for notebook compatibility

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM
import shutil

OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Results directory with timestamped run
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# # Setup for API clients
# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False
#     logger.warning("OpenAI package not found. Install with: pip install openai")

try:
    import dotenv
    dotenv.load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    logger.warning("python-dotenv not found. Install with: pip install python-dotenv")

# Constants
# DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MODEL = "qwen3-instruct-2507:4b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500


# Helper Functions
# ================

def setup_client(api_key=None, model=DEFAULT_MODEL):
    """
    Set up the API client for LLM interactions.

    Args:
        api_key: API key (if None, will look for OPENAI_API_KEY in env)
        model: Model name to use

    Returns:
        tuple: (client, model_name)
    """
    # if api_key is None:
    #     api_key = os.environ.get("OPENAI_API_KEY")
    #     if api_key is None and not ENV_LOADED:
    #         logger.warning("No API key found. Set OPENAI_API_KEY env var or pass api_key param.")
    
    # if OPENAI_AVAILABLE:
    #     client = OpenAI(api_key=api_key)
    #     return client, model
    # else:
    #     logger.error("OpenAI package required. Install with: pip install openai")
    #     return None, model
    client = LlamacppLLM(model=model, verbose=True)
    return client, model


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count tokens in text string using the appropriate tokenizer.

    Args:
        text: Text to tokenize
        model: Model name to use for tokenization

    Returns:
        int: Token count
    """
    # try:
    #     encoding = tiktoken.encoding_for_model(model)
    #     return len(encoding.encode(text))
    # except Exception as e:
    #     # Fallback for when tiktoken doesn't support the model
    #     logger.warning(f"Could not use tiktoken for {model}: {e}")
    #     # Rough approximation: 1 token ≈ 4 chars in English
    #     return len(text) // 4
    return token_counter(text, model=model)


def generate_response(
    prompt: str,
    client=None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_message: str = "You are a helpful assistant."
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a response from the LLM and return with metadata.

    Args:
        prompt: The prompt to send
        client: API client (if None, will create one)
        model: Model name
        temperature: Temperature parameter
        max_tokens: Maximum tokens to generate
        system_message: System message to use

    Returns:
        tuple: (response_text, metadata)
    """
    if client is None:
        client, model = setup_client(model=model)
        if client is None:
            return "ERROR: No API client available", {"error": "No API client"}
    
    prompt_tokens = count_tokens(prompt, model)
    system_tokens = count_tokens(system_message, model)
    
    metadata = {
        "prompt_tokens": prompt_tokens,
        "system_tokens": system_tokens,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timestamp": time.time()
    }
    
    try:
        # start_time = time.time()
        # response = client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_message},
        #         {"role": "user", "content": prompt}
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_tokens
        # )
        # latency = time.time() - start_time
        
        # response_text = response.choices[0].message.content

        start_time = time.time()
        response_stream = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            stream=True
        )

        response = ""
        for chunk in response_stream:
            response += chunk
        latency = time.time() - start_time

        response_text = response
        response_tokens = count_tokens(response_text, model)
        
        metadata.update({
            "latency": latency,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + system_tokens + response_tokens,
            "token_efficiency": response_tokens / (prompt_tokens + system_tokens) if (prompt_tokens + system_tokens) > 0 else 0,
            "tokens_per_second": response_tokens / latency if latency > 0 else 0
        })
        
        return response_text, metadata
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        metadata["error"] = str(e)
        return f"ERROR: {str(e)}", metadata


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary into a readable string.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        str: Formatted metrics string
    """
    # Select the most important metrics to show
    key_metrics = {
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "response_tokens": metrics.get("response_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "latency": f"{metrics.get('latency', 0):.2f}s",
        "token_efficiency": f"{metrics.get('token_efficiency', 0):.2f}"
    }
    
    return " | ".join([f"{k}: {v}" for k, v in key_metrics.items()])


# --- Helper: Save text content ---
def _save_text(content: str, filepath: pathlib.Path) -> None:
    """Save string content to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")


# --- Helper: Save plot ---
def _save_plot(fig: plt.Figure, filepath: pathlib.Path) -> None:
    """Save matplotlib figure to PNG and close."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)


# Keep original display_response for notebook use (optional)
def display_response(
    prompt: str,
    response: str,
    metrics: Dict[str, Any],
    show_prompt: bool = True
) -> None:
    """Display in notebook (no-op in script mode)."""
    if "IPython" not in globals():
        return
    if show_prompt:
        display(HTML("<h4>Prompt:</h4>"))
        display(Markdown(f"```\n{prompt}\n```"))
    display(HTML("<h4>Response:</h4>"))
    display(Markdown(response))
    display(HTML("<h4>Metrics:</h4>"))
    display(Markdown(f"```\n{format_metrics(metrics)}\n```"))


# Control Loop Base Classes
# =========================

class ControlLoop:
    """
    Base class for all control loop implementations.
    Provides common functionality for tracking metrics and history.
    """
    
    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        system_message: str = "You are a helpful assistant.",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        verbose: bool = False
    ):
        """
        Initialize the control loop.
        
        Args:
            client: API client (if None, will create one)
            model: Model name to use
            system_message: System message to use
            max_tokens: Maximum tokens to generate
            temperature: Temperature parameter
            verbose: Whether to print debug information
        """
        self.client, self.model = setup_client(model=model) if client is None else (client, model)
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize history and metrics tracking
        self.history = []
        self.metrics = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "steps": 0
        }
    
    def _log(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.
        
        Args:
            message: Message to log
        """
        if self.verbose:
            logger.info(message)
    
    def _call_llm(
        self,
        prompt: str,
        custom_system_message: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Call the LLM and update metrics.
        
        Args:
            prompt: Prompt to send
            custom_system_message: Override system message (optional)
            
        Returns:
            tuple: (response_text, metadata)
        """
        system_msg = custom_system_message if custom_system_message else self.system_message
        
        response, metadata = generate_response(
            prompt=prompt,
            client=self.client,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_message=system_msg
        )
        
        # Update metrics
        self.metrics["total_prompt_tokens"] += metadata.get("prompt_tokens", 0)
        self.metrics["total_response_tokens"] += metadata.get("response_tokens", 0)
        self.metrics["total_tokens"] += metadata.get("total_tokens", 0)
        self.metrics["total_latency"] += metadata.get("latency", 0)
        self.metrics["steps"] += 1
        
        # Add to history
        step_record = {
            "prompt": prompt,
            "response": response,
            "metrics": metadata,
            "timestamp": time.time()
        }
        self.history.append(step_record)
        
        return response, metadata
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics for all steps.
        
        Returns:
            dict: Summary metrics
        """
        summary = self.metrics.copy()
        
        # Add derived metrics
        if summary["steps"] > 0:
            summary["avg_latency_per_step"] = summary["total_latency"] / summary["steps"]
            
        if summary["total_prompt_tokens"] > 0:
            summary["overall_efficiency"] = (
                summary["total_response_tokens"] / summary["total_prompt_tokens"]
            )
        
        return summary
    
    def visualize_metrics(self) -> None:
        """Save metrics visualization to PNG in results directory."""
        if not self.history:
            logger.warning("No history to visualize")
            return

        steps = list(range(1, len(self.history) + 1))
        prompt_tokens = [h["metrics"].get("prompt_tokens", 0) for h in self.history]
        response_tokens = [h["metrics"].get("response_tokens", 0) for h in self.history]
        latencies = [h["metrics"].get("latency", 0) for h in self.history]
        efficiencies = [h["metrics"].get("token_efficiency", 0) for h in self.history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Control Loop Metrics by Step", fontsize=16)

        # Plot 1
        axes[0, 0].bar(steps, prompt_tokens, label="Prompt Tokens", color="blue", alpha=0.7)
        axes[0, 0].bar(steps, response_tokens, bottom=prompt_tokens, label="Response Tokens", color="green", alpha=0.7)
        axes[0, 0].set_title("Token Usage")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Tokens")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2
        axes[0, 1].plot(steps, latencies, marker='o', color="red", alpha=0.7)
        axes[0, 1].set_title("Latency")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].grid(alpha=0.3)

        # Plot 3
        axes[1, 0].plot(steps, efficiencies, marker='s', color="purple", alpha=0.7)
        axes[1, 0].set_title("Token Efficiency (Response/Prompt)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Ratio")
        axes[1, 0].grid(alpha=0.3)

        # Plot 4
        cumulative_tokens = np.cumsum([h["metrics"].get("total_tokens", 0) for h in self.history])
        axes[1, 1].plot(steps, cumulative_tokens, marker='^', color="orange", alpha=0.7)
        axes[1, 1].set_title("Cumulative Token Usage")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Total Tokens")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        plot_path = RUN_DIR / f"metrics_{self.__class__.__name__.lower()}.png"
        _save_plot(fig, plot_path)
        logger.info(f"Metrics plot saved: {plot_path}")



class SequentialChain(ControlLoop):
    """
    A control loop that chains multiple steps in sequence,
    where each step's output becomes input to the next step.
    """
    
    def __init__(self, steps: List[Dict[str, Any]], **kwargs):
        """
        Initialize the sequential chain.
        
        Args:
            steps: List of step configurations, each with:
                - prompt_template: str with {input} placeholder
                - system_message: (optional) custom system message
                - name: (optional) step name
            **kwargs: Additional args passed to ControlLoop
        """
        super().__init__(**kwargs)
        self.steps = steps
        self._validate_steps()
    
    def _validate_steps(self) -> None:
        """Validate step configurations."""
        for i, step in enumerate(self.steps):
            if "prompt_template" not in step:
                raise ValueError(f"Step {i} missing 'prompt_template'")
            
            # Ensure each step has a name
            if "name" not in step:
                step["name"] = f"step_{i+1}"
    
    def run(self, initial_input: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the sequential chain with the given initial input.
        
        Args:
            initial_input: The input to the first step
            
        Returns:
            tuple: (final_output, all_outputs)
        """
        current_input = initial_input
        all_outputs = {"initial_input": initial_input}
        
        for i, step in enumerate(self.steps):
            step_name = step["name"]
            self._log(f"Running step {i+1}/{len(self.steps)}: {step_name}")
            
            # Format prompt using current input
            prompt = step["prompt_template"].format(input=current_input)
            system_message = step.get("system_message", self.system_message)
            
            # Call LLM
            response, metadata = self._call_llm(prompt, system_message)
            
            # Store output
            all_outputs[step_name] = {
                "prompt": prompt,
                "response": response,
                "metrics": metadata
            }
            
            # Update input for next step
            current_input = response
        
        return current_input, all_outputs

    def save_chain_results(self, all_outputs: Dict[str, Any], example_name: str) -> None:
        """Save all chain results to disk."""
        base = RUN_DIR / example_name / "sequential_chain"
        base.mkdir(parents=True, exist_ok=True)

        # Initial input
        _save_text(all_outputs["initial_input"], base / "00_initial_input.txt")

        # Each step
        for i, step in enumerate(self.steps):
            step_name = step["name"]
            data = all_outputs[step_name]
            step_dir = base / f"{i+1:02d}_{step_name}"
            step_dir.mkdir(exist_ok=True)
            _save_text(data["prompt"], step_dir / "prompt.txt")
            _save_text(data["response"], step_dir / "response.txt")
            _save_text(format_metrics(data["metrics"]), step_dir / "metrics.txt")

        # Summary
        summary = self.get_summary_metrics()
        summary_text = f"""

Total Steps: {summary['steps']}
Total Tokens: {summary['total_tokens']}
Total Latency: {summary['total_latency']:.2f}s
Avg. Latency per Step: {summary.get('avg_latency_per_step', 0):.2f}s
Overall Efficiency: {summary.get('overall_efficiency', 0):.2f}
"""
        _save_text(summary_text.strip(), base / "summary.txt")

        # Full JSON
        with open(base / "full_results.json", "w", encoding="utf-8") as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)


class IterativeRefiner(ControlLoop):
    """
    A control loop that iteratively refines an output through multiple cycles
    of feedback and improvement until a stopping condition is met.
    """
    
    def __init__(
        self,
        max_iterations: int = 5,
        refinement_template: str = "Please improve the following text: {previous_response}\n\nSpecific improvements needed: {feedback}",
        feedback_template: str = "Evaluate the quality of this response and suggest specific improvements: {response}",
        stopping_condition: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        **kwargs
    ):
        """
        Initialize the iterative refiner.
        
        Args:
            max_iterations: Maximum number of refinement iterations
            refinement_template: Template for refinement prompts
            feedback_template: Template for generating feedback
            stopping_condition: Function that takes (response, metadata) and returns
                               True if refinement should stop
            **kwargs: Additional args passed to ControlLoop
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.refinement_template = refinement_template
        self.feedback_template = feedback_template
        self.stopping_condition = stopping_condition
    
    def generate_feedback(self, response: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generate feedback on the current response.
        
        Args:
            response: Current response to evaluate
            
        Returns:
            tuple: (feedback, metadata)
        """
        prompt = self.feedback_template.format(response=response)
        return self._call_llm(prompt)
    
    def refine_response(
        self,
        previous_response: str,
        feedback: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Refine the response based on feedback.
        
        Args:
            previous_response: Previous response to refine
            feedback: Feedback to use for refinement
            
        Returns:
            tuple: (refined_response, metadata)
        """
        prompt = self.refinement_template.format(
            previous_response=previous_response,
            feedback=feedback
        )
        return self._call_llm(prompt)
    
    def run(
        self,
        initial_prompt: str,
        use_auto_feedback: bool = True
    ) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Run the iterative refinement process.
        
        Args:
            initial_prompt: Initial prompt to generate first response
            use_auto_feedback: Whether to auto-generate feedback (if False,
                              you need to provide feedback manually)
                              
        Returns:
            tuple: (final_response, refinement_history)
        """
        # Generate initial response
        self._log("Generating initial response")
        current_response, metadata = self._call_llm(initial_prompt)
        
        refinement_history = {
            "initial": {
                "prompt": initial_prompt,
                "response": current_response,
                "metrics": metadata
            },
            "iterations": []
        }
        
        # Iterative refinement loop
        iteration = 0
        should_continue = True
        
        while should_continue and iteration < self.max_iterations:
            iteration += 1
            self._log(f"Refinement iteration {iteration}/{self.max_iterations}")
            
            # Generate feedback
            if use_auto_feedback:
                feedback, feedback_metadata = self.generate_feedback(current_response)
                self._log(f"Auto-feedback: {feedback}")
            else:
                # Manual feedback mode
                print(f"\n\nCurrent response (iteration {iteration}):")
                print("-" * 80)
                print(current_response)
                print("-" * 80)
                feedback = input("Enter your feedback (or 'stop' to end refinement): ")
                
                if feedback.lower() == 'stop':
                    break
                
                feedback_metadata = {"manual": True}
            
            # Refine response
            refined_response, refine_metadata = self.refine_response(current_response, feedback)
            
            # Record iteration
            refinement_history["iterations"].append({
                "iteration": iteration,
                "feedback": feedback,
                "feedback_metrics": feedback_metadata,
                "refined_response": refined_response,
                "refinement_metrics": refine_metadata
            })
            
            # Update current response
            current_response = refined_response
            
            # Check stopping condition
            if self.stopping_condition:
                should_continue = not self.stopping_condition(current_response, refine_metadata)
        
        return current_response, refinement_history

    def save_refinement_history(self, refinement_history: Dict[str, Any], example_name: str) -> None:
        """Save refinement history to disk."""
        base = RUN_DIR / example_name / "iterative_refiner"
        base.mkdir(parents=True, exist_ok=True)

        init = refinement_history["initial"]
        _save_text(init["prompt"], base / "00_initial_prompt.txt")
        _save_text(init["response"], base / "00_initial_response.txt")
        _save_text(format_metrics(init["metrics"]), base / "00_initial_metrics.txt")

        for it in refinement_history["iterations"]:
            it_dir = base / f"iteration_{it['iteration']:02d}"
            it_dir.mkdir(exist_ok=True)
            _save_text(it["feedback"], it_dir / "feedback.txt")
            _save_text(it["refined_response"], it_dir / "response.txt")
            _save_text(format_metrics(it["refinement_metrics"]), it_dir / "metrics.txt")

        total_iters = len(refinement_history["iterations"])
        final_tokens = (
            refinement_history["iterations"][-1]["refinement_metrics"]["response_tokens"]
            if total_iters > 0 else init["metrics"]["response_tokens"]
        )
        summary_text = f"""

Initial prompt tokens: {init['metrics']['prompt_tokens']}
Initial response tokens: {init['metrics']['response_tokens']}
Total refinement iterations: {total_iters}
Final response tokens: {final_tokens}
"""
        _save_text(summary_text.strip(), base / "summary.txt")

        with open(base / "full_history.json", "w", encoding="utf-8") as f:
            json.dump(refinement_history, f, indent=2, ensure_ascii=False)



class ConditionalBrancher(ControlLoop):
    """
    A control loop that implements conditional branching based on LLM outputs,
    allowing for different execution paths depending on conditions.
    """
    
    def __init__(
        self,
        branches: Dict[str, Dict[str, Any]],
        classifier_template: str = "Analyze the following input and classify it into exactly one of these categories: {categories}.\n\nInput: {input}\n\nCategory:",
        **kwargs
    ):
        """
        Initialize the conditional brancher.
        
        Args:
            branches: Dictionary mapping branch names to configurations:
                - prompt_template: str with {input} placeholder
                - system_message: (optional) custom system message
            classifier_template: Template for classification prompt
            **kwargs: Additional args passed to ControlLoop
        """
        super().__init__(**kwargs)
        self.branches = branches
        self.classifier_template = classifier_template
        self._validate_branches()
    
    def _validate_branches(self) -> None:
        """Validate branch configurations."""
        if not self.branches:
            raise ValueError("No branches defined")
        
        for branch_name, config in self.branches.items():
            if "prompt_template" not in config:
                raise ValueError(f"Branch '{branch_name}' missing 'prompt_template'")
    
    def classify_input(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify input to determine which branch to take.
        
        Args:
            input_text: Input text to classify
            
        Returns:
            tuple: (branch_name, metadata)
        """
        categories = list(self.branches.keys())
        categories_str = ", ".join(categories)
        
        prompt = self.classifier_template.format(
            categories=categories_str,
            input=input_text
        )
        
        # Use a specific system message for classification
        system_message = "You are a classifier that categorizes inputs precisely and accurately."
        response, metadata = self._call_llm(prompt, system_message)
        
        # Extract the branch name from the response
        # First try to match a category exactly
        for category in categories:
            if category.lower() in response.lower():
                return category, metadata
        
        # If no exact match, take the first line as the response and find closest match
        first_line = response.strip().split('\n')[0].lower()
        
        best_match = None
        best_score = 0
        
        for category in categories:
            # Simple string similarity score
            cat_lower = category.lower()
            matches = sum(c in first_line for c in cat_lower)
            score = matches / len(cat_lower) if len(cat_lower) > 0 else 0
            
            if score > best_score:
                best_score = score
                best_match = category
        
        if best_match and best_score > 0.5:
            return best_match, metadata
        
        # Fallback to first category if no match found
        self._log(f"Warning: Could not classify input. Using first branch: {categories[0]}")
        return categories[0], metadata
    
    def execute_branch(
        self,
        branch_name: str,
        input_text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute a specific branch with the given input.
        
        Args:
            branch_name: Name of branch to execute
            input_text: Input text for the branch
            
        Returns:
            tuple: (response, metadata)
        """
        if branch_name not in self.branches:
            raise ValueError(f"Unknown branch: {branch_name}")
        
        branch_config = self.branches[branch_name]
        prompt = branch_config["prompt_template"].format(input=input_text)
        system_message = branch_config.get("system_message", self.system_message)
        
        return self._call_llm(prompt, system_message)
    
    def run(
        self,
        input_text: str,
        branch_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run the conditional branching process.
        
        Args:
            input_text: Input text to process
            branch_name: Optional branch to use (skips classification)
            
        Returns:
            tuple: (response, run_details)
        """
        run_details = {"input": input_text}
        
        # Classify input if branch not specified
        if branch_name is None:
            self._log("Classifying input")
            branch_name, classification_metadata = self.classify_input(input_text)
            run_details["classification"] = {
                "branch": branch_name,
                "metrics": classification_metadata
            }
        
        self._log(f"Executing branch: {branch_name}")
        
        # Execute selected branch
        response, metadata = self.execute_branch(branch_name, input_text)
        
        run_details["execution"] = {
            "branch": branch_name,
            "response": response,
            "metrics": metadata
        }
        
        return response, run_details

    def save_branching_results(self, run_details: Dict[str, Any], query: str, example_name: str) -> None:
        """Save branching results per query."""
        safe_query = re.sub(r"[^\w\-_.]", "_", query)[:50]
        base = RUN_DIR / example_name / "conditional_brancher" / safe_query
        base.mkdir(parents=True, exist_ok=True)

        _save_text(run_details["input"], base / "input.txt")

        if "classification" in run_details:
            cls = run_details["classification"]
            _save_text(cls["branch"], base / "classified_branch.txt")
            _save_text(format_metrics(cls["metrics"]), base / "classification_metrics.txt")

        exec_data = run_details["execution"]
        _save_text(exec_data["branch"], base / "executed_branch.txt")
        _save_text(exec_data["response"], base / "response.txt")
        _save_text(format_metrics(exec_data["metrics"]), base / "execution_metrics.txt")

        with open(base / "full_details.json", "w", encoding="utf-8") as f:
            json.dump(run_details, f, indent=2, ensure_ascii=False)


class SelfCritique(ControlLoop):
    """
    A control loop that generates a response, then critiques and improves it
    in a single flow, without requiring multiple API calls for refinement.
    """
    
    def __init__(
        self,
        critique_template: str = "Step 1: Generate a response to the question.\nStep 2: Critique your response for any errors, omissions, or improvements.\nStep 3: Provide a final, improved response based on your critique.\n\nQuestion: {input}",
        parse_sections: bool = True,
        **kwargs
    ):
        """
        Initialize the self-critique control loop.
        
        Args:
            critique_template: Template for the self-critique prompt
            parse_sections: Whether to parse the response into sections
            **kwargs: Additional args passed to ControlLoop
        """
        super().__init__(**kwargs)
        self.critique_template = critique_template
        self.parse_sections = parse_sections
    
    def run(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the self-critique process.
        
        Args:
            input_text: Input to respond to
            
        Returns:
            tuple: (final_response, run_details)
        """
        # Format prompt
        prompt = self.critique_template.format(input=input_text)
        
        # Generate self-critique response
        response, metadata = self._call_llm(prompt)
        
        # Parse sections if requested
        sections = {}
        if self.parse_sections:
            # Attempt to parse initial response, critique, and final response
            initial_match = re.search(r"Step 1:(.*?)Step 2:", response, re.DOTALL)
            critique_match = re.search(r"Step 2:(.*?)Step 3:", response, re.DOTALL)
            final_match = re.search(r"Step 3:(.*?)$", response, re.DOTALL)
            
            if initial_match:
                sections["initial_response"] = initial_match.group(1).strip()
            if critique_match:
                sections["critique"] = critique_match.group(1).strip()
            if final_match:
                sections["final_response"] = final_match.group(1).strip()
        
        # If parsing failed, use the full response
        if not sections and self.parse_sections:
            self._log("Failed to parse sections from response")
            sections["full_response"] = response
        
        # Create run details
        run_details = {
            "input": input_text,
            "full_response": response,
            "sections": sections,
            "metrics": metadata
        }
        
        # Return final response (or full response if parsing failed)
        final_response = sections.get("final_response", response)
        return final_response, run_details

    def save_results(self, run_details: Dict[str, Any], example_name: str) -> None:
        """Save self-critique results."""
        base = RUN_DIR / example_name / "self_critique"
        base.mkdir(parents=True, exist_ok=True)

        _save_text(run_details["input"], base / "input.txt")
        _save_text(run_details["full_response"], base / "full_response.txt")

        sections = run_details.get("sections", {})
        if "initial_response" in sections:
            _save_text(sections["initial_response"], base / "initial_response.txt")
        if "critique" in sections:
            _save_text(sections["critique"], base / "critique.txt")
        if "final_response" in sections:
            _save_text(sections["final_response"], base / "final_response.txt")
        elif "full_response" in sections:
            _save_text(sections["full_response"], base / "final_response.txt")

        _save_text(format_metrics(run_details["metrics"]), base / "metrics.txt")

        with open(base / "full_details.json", "w", encoding="utf-8") as f:
            json.dump(run_details, f, indent=2, ensure_ascii=False)


class ExternalValidation(ControlLoop):
    """
    A control loop that uses external tools or knowledge to validate
    and correct LLM responses, creating a closed feedback loop.
    """
    
    def __init__(
        self,
        validator_fn: Callable[[str], Tuple[bool, str]],
        correction_template: str = "Your previous response had some issues:\n\n{validation_feedback}\n\nPlease correct your response to address these issues:\n\n{previous_response}",
        max_attempts: int = 3,
        **kwargs
    ):
        """
        Initialize the external validation loop.
        
        Args:
            validator_fn: Function that takes a response and returns
                        (is_valid, feedback_message)
            correction_template: Template for correction prompts
            max_attempts: Maximum validation attempts
            **kwargs: Additional args passed to ControlLoop
        """
        super().__init__(**kwargs)
        self.validator_fn = validator_fn
        self.correction_template = correction_template
        self.max_attempts = max_attempts
    
    def run(self, input_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Run the external validation process.
        
        Args:
            input_text: Input to respond to
            
        Returns:
            tuple: (final_response, run_details)
        """
        # Generate initial response
        response, metadata = self._call_llm(input_text)
        
        attempts = []
        current_response = response
        is_valid = False
        validation_feedback = ""
        
        # Add initial attempt
        attempts.append({
            "attempt": 1,
            "response": current_response,
            "metrics": metadata,
            "validation": {
                "pending": True
            }
        })
        
        # Validation loop
        for attempt in range(1, self.max_attempts + 1):
            # Validate the current response
            self._log(f"Validating attempt {attempt}")
            is_valid, validation_feedback = self.validator_fn(current_response)
            
            # Update validation results for the current attempt
            attempts[-1]["validation"] = {
                "is_valid": is_valid,
                "feedback": validation_feedback,
                "pending": False
            }
            
            # Stop if valid
            if is_valid:
                self._log(f"Valid response on attempt {attempt}")
                break
            
            # Stop if max attempts reached
            if attempt >= self.max_attempts:
                self._log(f"Max attempts ({self.max_attempts}) reached without valid response")
                break
            
            # Create correction prompt
            self._log(f"Attempting correction (attempt {attempt+1})")
            correction_prompt = self.correction_template.format(
                validation_feedback=validation_feedback,
                previous_response=current_response
            )
            
            # Generate corrected response
            corrected_response, correction_metadata = self._call_llm(correction_prompt)
            current_response = corrected_response
            
            # Add new attempt
            attempts.append({
                "attempt": attempt + 1,
                "response": current_response,
                "metrics": correction_metadata,
                "validation": {
                    "pending": True
                }
            })
        
        # Create run details
        run_details = {
            "input": input_text,
            "attempts": attempts,
            "final_response": current_response,
            "is_valid": is_valid,
            "validation_feedback": validation_feedback,
            "attempts_count": len(attempts)
        }
        
        return current_response, run_details

    def save_results(self, run_details: Dict[str, Any], example_name: str) -> None:
        """Save validation loop results."""
        base = RUN_DIR / example_name / "external_validation"
        base.mkdir(parents=True, exist_ok=True)

        _save_text(run_details["input"], base / "input.txt")

        for attempt in run_details["attempts"]:
            a_dir = base / f"attempt_{attempt['attempt']:02d}"
            a_dir.mkdir(exist_ok=True)
            _save_text(attempt["response"], a_dir / "response.txt")
            _save_text(format_metrics(attempt["metrics"]), a_dir / "metrics.txt")

            val = attempt["validation"]
            if not val["pending"]:
                status = "VALID" if val["is_valid"] else "INVALID"
                _save_text(status, a_dir / "validation_status.txt")
                if not val["is_valid"]:
                    _save_text(val["feedback"], a_dir / "validation_feedback.txt")

        summary_text = f"""

Final status: {'VALID' if run_details['is_valid'] else 'INVALID'}
Total attempts: {run_details['attempts_count']}
Total tokens: {self.metrics['total_tokens']}
Total latency: {self.metrics['total_latency']:.2f}s
"""
        _save_text(summary_text.strip(), base / "summary.txt")

        with open(base / "full_details.json", "w", encoding="utf-8") as f:
            json.dump(run_details, f, indent=2, ensure_ascii=False)


# Example Usage
# =============

def example_sequential_chain():
    """Example of a sequential chain for data analysis."""
    steps = [
        {
            "name": "extract_entities",
            "prompt_template": "Extract the main entities (people, places, organizations) from this text. For each entity, provide a brief description.\n\nText: {input}",
            "system_message": "You are an expert at extracting and categorizing named entities from text."
        },
        {
            "name": "analyze_relationships",
            "prompt_template": "Based on these entities, analyze the relationships between them:\n\n{input}",
            "system_message": "You are an expert at analyzing relationships between entities."
        },
        {
            "name": "generate_report",
            "prompt_template": "Create a concise summary report based on this relationship analysis:\n\n{input}",
            "system_message": "You are an expert at creating clear, concise reports."
        }
    ]
    
    chain = SequentialChain(steps=steps, verbose=True)
    
    sample_text = """
    In 1995, Jeff Bezos founded Amazon in Seattle. Initially an online bookstore, 
    Amazon expanded rapidly under Bezos' leadership. By 2021, Amazon had become 
    one of the world's most valuable companies, and Bezos had briefly overtaken 
    Elon Musk as the world's richest person. Musk, the CEO of Tesla and SpaceX, 
    later reclaimed the top spot after Tesla's stock surged. Meanwhile, Microsoft, 
    founded by Bill Gates in Albuquerque in 1975, continued to be a major tech 
    competitor under CEO Satya Nadella.
    """
    
    final_output, all_outputs = chain.run(sample_text)
    chain.save_chain_results(all_outputs, "example_01_sequential")
    chain.visualize_metrics()
    return final_output, all_outputs


def example_iterative_refiner():
    """Example of iterative refinement for essay writing."""
    # Define a stopping condition based on a quality threshold
    def quality_threshold(response, metadata):
        # Stop if response is over 500 tokens and latency is acceptable
        response_tokens = metadata.get("response_tokens", 0)
        latency = metadata.get("latency", 0)
        return response_tokens > 500 and latency < 5.0
    
    refiner = IterativeRefiner(
        max_iterations=3,
        stopping_condition=quality_threshold,
        verbose=True
    )
    
    prompt = "Write a short essay on the future of artificial intelligence."
    
    final_response, refinement_history = refiner.run(prompt)
    refiner.save_refinement_history(refinement_history, "example_02_refiner")
    refiner.visualize_metrics()
    return final_response, refinement_history


def example_conditional_brancher():
    """Example of conditional branching for query routing."""
    branches = {
        "technical": {
            "prompt_template": "Provide a technical, detailed explanation of this topic for an expert audience:\n\n{input}",
            "system_message": "You are a technical expert who provides detailed, precise explanations."
        },
        "simplified": {
            "prompt_template": "Explain this topic in simple terms that a 10-year-old would understand:\n\n{input}",
            "system_message": "You are an educator who explains complex topics in simple, accessible language."
        },
        "practical": {
            "prompt_template": "Provide practical, actionable advice on this topic:\n\n{input}",
            "system_message": "You are a practical advisor who provides concrete, actionable guidance."
        }
    }
    
    brancher = ConditionalBrancher(branches=branches, verbose=True)
    
    queries = [
        "How does quantum computing work?",
        "What is climate change?",
        "How can I improve my public speaking skills?"
    ]
    
    results = []
    for i, query in enumerate(queries):
        response, run_details = brancher.run(query)
        brancher.save_branching_results(run_details, query, f"example_03_brancher/query_{i+1:02d}")
        results.append((query, response, run_details))
    brancher.visualize_metrics()
    return results


def example_self_critique():
    """Example of self-critique for fact-checking."""
    critique = SelfCritique(
        critique_template="""
        Answer the following question with factual information:
        
        Question: {input}
        
        Step 1: Write an initial response with all the information you think is relevant.
        
        Step 2: Critically review your response. Check for:
        - Factual errors or inaccuracies
        - Missing important information
        - Potential biases or one-sided perspectives
        - Areas where you're uncertain and should express less confidence
        
        Step 3: Write an improved final response that addresses the issues identified in your critique.
        """,
        verbose=True
    )
    
    query = "What were the major causes of World War I and how did they lead to the conflict?"
    
    final_response, run_details = critique.run(query)
    critique.save_results(run_details, "example_04_selfcritique")
    critique.visualize_metrics()
    return final_response, run_details


def example_external_validation():
    """Example of external validation for code generation."""
    # Simple validator function that checks for Python syntax errors
    def python_validator(code_response):
        # Extract code blocks
        import re
        code_blocks = re.findall(r"```python(.*?)```", code_response, re.DOTALL)
        
        if not code_blocks:
            return False, "No Python code blocks found in the response."
        
        # Check each block for syntax errors
        for i, block in enumerate(code_blocks):
            try:
                compile(block, "<string>", "exec")
            except SyntaxError as e:
                return False, f"Syntax error in code block {i+1}: {str(e)}"
        
        return True, "Code syntax is valid."
    
    validator = ExternalValidation(
        validator_fn=python_validator,
        max_attempts=3,
        verbose=True
    )
    
    prompt = "Write a Python function to check if a string is a palindrome."
    
    final_response, run_details = validator.run(prompt)
    validator.save_results(run_details, "example_05_validation")
    validator.visualize_metrics()
    return final_response, run_details


# Main execution (when run as a script)
if __name__ == "__main__":
    print(f"Running all examples → {RUN_DIR}")
    print("="*60)

    # Example 1: Sequential Chain
    print("="*60)
    print("EXAMPLE 1: Sequential Chain (Entity → Relationships → Report)")
    print("="*60)
    example_sequential_chain()

    # Example 2: Iterative Refiner
    print("="*60)
    print("EXAMPLE 2: Iterative Refiner (Essay on AI Future)")
    print("="*60)
    example_iterative_refiner()

    # Example 3: Conditional Brancher
    print("="*60)
    print("EXAMPLE 3: Conditional Brancher (Query Routing)")
    print("="*60)
    example_conditional_brancher()

    # Example 4: Self-Critique
    print("="*60)
    print("EXAMPLE 4: Self-Critique (WW1 Causes)")
    print("="*60)
    example_self_critique()

    # Example 5: External Validation
    print("="*60)
    print("EXAMPLE 5: External Validation (Palindrome Function)")
    print("="*60)
    example_external_validation()

    print("="*60)
    print(f"ALL RESULTS SAVED TO: {RUN_DIR}")
    print("Structure:")
    print("  ├── example_01_sequential/")
    print("  ├── example_02_refiner/")
    print("  ├── example_03_brancher/query_01_.../")
    print("  ├── example_04_selfcritique/")
    print("  ├── example_05_validation/")
    print("  └── metrics_*.png")
    print("="*60)
