#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context-Engineering: Recursive Patterns for Self-Improving Contexts
==================================================================

This module explores recursive patterns in context engineering - approaches
that enable LLMs to extend, refine, and evolve their own context. These patterns
create feedback loops within prompts, allowing for iterative improvement,
self-verification, and emergent capabilities beyond what's explicitly coded.

Key concepts covered:
1. Basic recursive patterns (self-reflection, bootstrapping)
2. Field protocols and shells as recursive frameworks
3. Symbolic residue and state tracking
4. Boundary collapse and gradient systems
5. Emergent attractors and resonance

Usage:
    # In Jupyter or Colab:
    %run 07_recursive_patterns.py
    # or
    from recursive_patterns import RecursivePattern, FieldProtocol, SymbolicResidue
"""

import os
import re
import json
import time
import uuid
import hashlib
import logging
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, TypeVar, Set
from IPython.display import display, Markdown, HTML, JSON

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM
import shutil
import pathlib

from pydantic import BaseModel
from pydantic import __version__ as pydantic_version

OUTPUT_ROOT = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for required libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not found. Install with: pip install openai")

try:
    import dotenv
    dotenv.load_dotenv()
    ENV_LOADED = True
except ImportError:
    ENV_LOADED = False
    logger.warning("python-dotenv not found. Install with: pip install python-dotenv")

# Constants
DEFAULT_MODEL = "qwen3-instruct-2507:4b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# --- Add structured response models ---

class SelfReflectionResponse(BaseModel):
    response: str
    improvement_score: float = 0.0

class BootstrappingResponse(BaseModel):
    approach: str

class ResidueResponse(BaseModel):
    output: str
    symbolic_residue: List[str] = []
    residue_compression: float = 0.0
    resonance_score: float = 0.0

# Helper Functions
# ===============

def setup_client(api_key=None, model=DEFAULT_MODEL):
    client = LlamacppLLM(model=model, verbose=True)
    return client, model

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    return token_counter(text, model=model)

def _safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Serialize Pydantic models, dataclasses, or any object with .dict()/.model_dump()
    while falling back to str() for unknown types.
    """
    if isinstance(obj, str):
        return obj
    try:
        # Pydantic v2
        if hasattr(obj, "model_dump"):
            return json.dumps(obj.model_dump(), indent=indent, ensure_ascii=False)
        # Pydantic v1
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), indent=indent, ensure_ascii=False)
    except Exception:
        pass
    # Last resort: try built-in json on dict representation
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"__error__": "Object not serializable", "__repr__": repr(obj)}, indent=indent)

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

def save_recursive_pattern(
    example_dir: pathlib.Path,
    pattern_name: str,
    input_data: Any,
    iterations: List[Dict[str, Any]],
    final_output: Any,
    metrics: Dict[str, Any] = None
) -> None:
    """
    Save full recursive pattern execution to `example_dir` as HTML + JSON.

    Args:
        pattern_name: Name of the recursive pattern
        input_data: Initial input data
        iterations: List of iteration data
        final_output: Final output data
        metrics: Optional metrics dictionary
    """
    example_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSONs
    (example_dir / "input.json").write_text(_safe_json_dumps(input_data))
    (example_dir / "final_output.json").write_text(_safe_json_dumps(final_output))
    if metrics:
        (example_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2))

    # Build HTML
    html_lines = [f"<h2>Recursive Pattern: {pattern_name}</h2>"]
    html_lines += ["<h3>Initial Input</h3>", "<pre><code>"]
    if isinstance(input_data, str):
        html_lines.append(input_data)
    else:
        html_lines.append(json.dumps(input_data, indent=2))
    html_lines += ["</code></pre>", "<h3>Recursive Iterations</h3>"]

    for i, it in enumerate(iterations):
        html_lines.append(f"<h4>Iteration {i+1}</h4>")
        if "prompt" in it:
            html_lines += ["<p><strong>Prompt:</strong></p>", "<pre><code>"]
            html_lines.append(it["prompt"])
            html_lines += ["</code></pre>"]
        if "response" in it:
            html_lines += ["<p><strong>Response:</strong></p>", "<pre><code>"]
            html_lines.append(it["response"])
            html_lines += ["</code></pre>"]
        if "state" in it:
            html_lines += ["<p><strong>State:</strong></p>", "<pre><code class='language-json'>"]
            state_json = _safe_json_dumps(it["state"])
            html_lines.append(state_json)
            html_lines += ["</code></pre>"]
        if "metrics" in it:
            html_lines += ["<p><strong>Metrics:</strong></p>", "<pre><code>"]
            html_lines.append(format_metrics(it["metrics"]))
            html_lines += ["</code></pre>"]

    html_lines += ["<h3>Final Output</h3>", "<pre><code>"]
    if isinstance(final_output, str):
        html_lines.append(final_output)
    elif isinstance(final_output, BaseModel):
        html_lines.append(json.dumps(final_output.model_json_schema(), indent=2))
    else:
        html_lines.append(json.dumps(final_output, indent=2))
    html_lines += ["</code></pre>"]

    if metrics:
        html_lines += ["<h3>Overall Metrics</h3>", "<pre><code>"]
        html_lines.append(format_metrics(metrics))
        html_lines += ["</code></pre>"]

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{pattern_name}</title>
<style>pre{{background:#f4f4f4;padding:1em;overflow:auto;}} code{{font-family:monospace;}}</style>
<script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/prism.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css" rel="stylesheet"/>
</head><body>{''.join(html_lines)}</body></html>"""
    (example_dir / "report.html").write_text(full_html)

    # Save per-iteration JSONs
    for i, it in enumerate(iterations):
        it_dir = example_dir / f"iteration_{i+1}"
        it_dir.mkdir(exist_ok=True)
        if "prompt" in it:
            (it_dir / "prompt.txt").write_text(it["prompt"])
        if "response" in it:
            (it_dir / "response.txt").write_text(it["response"])
        if "state" in it:
            (it_dir / "state.json").write_text(_safe_json_dumps(it["state"]))
        if "metrics" in it:
            (it_dir / "metrics.json").write_text(json.dumps(it["metrics"], indent=2))

    print(f"Saved recursive pattern '{pattern_name}' to: {example_dir}")

# =========================
# UPDATED BASE AND SUBCLASSES
# =========================

class RecursivePattern:
    """
    Base class for recursive patterns - approaches that enable LLMs
    to extend, refine, and evolve their own context.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        client=None,
        model: str = DEFAULT_MODEL,
        system_message: str = "You are a helpful assistant.",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        self.name = name
        self.description = description
        self.client, self.model = setup_client(model=model) if client is None else (client, model)
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose

        self.state = {}
        self.iterations = []

        self.metrics = {
            "total_prompt_tokens": 0,
            "total_response_tokens": 0,
            "total_tokens": 0,
            "total_latency": 0,
            "iterations": 0
        }

    def _log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def _generate_recursive_prompt(self, iteration: int, **kwargs) -> str:
        raise NotImplementedError("Subclasses must implement _generate_recursive_prompt")

    # --- Update: Use chat_structured when a response_model is specified ---
    def _call_llm(
        self,
        prompt: str,
        custom_system_message: Optional[str] = None,
        response_model: Optional[type[BaseModel]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        system_msg = custom_system_message or self.system_message
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        if response_model:
            response_stream = self.client.chat_structured_stream(
                messages=messages,
                response_model=response_model,
                temperature=self.temperature,
            )
            response_obj = list(response_stream)[0]
            response_text = str(response_obj)
        else:
            response_stream = self.client.chat_stream(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            chunks = list(response_stream)
            response_text = "".join(chunk.choices[0].delta.content for chunk in chunks)
            response_obj = response_text

        prompt_tokens = count_tokens(prompt, self.model)
        system_tokens = count_tokens(system_msg, self.model)
        response_tokens = count_tokens(response_text, self.model)
        metadata = {
            "prompt_tokens": prompt_tokens,
            "system_tokens": system_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + system_tokens + response_tokens,
            "latency": 0.0,  # Note: chat_structured does not expose latency
        }
        self.metrics["total_prompt_tokens"] += prompt_tokens
        self.metrics["total_response_tokens"] += response_tokens
        self.metrics["total_tokens"] += metadata["total_tokens"]
        self.metrics["iterations"] += 1
        return response_obj, metadata

    def _process_response(self, response: str, iteration: int) -> Any:
        # Default: just return the raw response
        return response

    def _update_state(
        self,
        iteration: int,
        prompt: str,
        response: Any,
        processed_output: Any,
        metrics: Dict[str, Any]
    ) -> None:
        iteration_record = {
            "iteration": iteration,
            "prompt": prompt,
            "response": str(response),
            "output": processed_output,
            "state": self.state.copy(),
            "metrics": metrics,
            "timestamp": time.time()
        }
        self.iterations.append(iteration_record)
        self.state["current_iteration"] = iteration
        self.state["last_prompt"] = prompt
        self.state["last_response"] = str(response)
        self.state["last_output"] = processed_output

    def _should_continue(self, iteration: int, current_output: Any) -> bool:
        return iteration < self.max_iterations

    def run(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        self.state = {"input": input_data}
        self.iterations = []

        self._log(f"Starting recursive pattern: {self.name}")

        current_output = input_data
        iteration = 0

        while True:
            iteration += 1
            self._log(f"Iteration {iteration}/{self.max_iterations}")
            prompt_kwargs = {"iteration": iteration, "current_output": current_output}
            prompt_kwargs.update(self.state)
            if "input" not in prompt_kwargs:
                prompt_kwargs["input"] = input_data
            prompt = self._generate_recursive_prompt(**prompt_kwargs)
            response, metrics = self._call_llm(prompt)
            processed_output = self._process_response(response, iteration)
            self._update_state(iteration, prompt, response, processed_output, metrics)
            current_output = processed_output
            if not self._should_continue(iteration, current_output):
                self._log(f"Stopping at iteration {iteration}")
                break
        return current_output, self.iterations

    def get_summary_metrics(self) -> Dict[str, Any]:
        summary = self.metrics.copy()
        if summary["iterations"] > 0:
            summary["avg_latency_per_iteration"] = float(summary["total_latency"]) / float(summary["iterations"])
        if summary["total_prompt_tokens"] > 0:
            summary["overall_efficiency"] = float(summary["total_response_tokens"]) / float(summary["total_prompt_tokens"])
        return summary

    def display_execution(self) -> None:
        pass

    def visualize_metrics(self) -> None:
        if not self.iterations:
            logger.warning("No iterations to visualize")
            return

        iterations = list(range(1, len(self.iterations) + 1))
        prompt_tokens = [it["metrics"].get("prompt_tokens", 0) for it in self.iterations]
        response_tokens = [it["metrics"].get("response_tokens", 0) for it in self.iterations]
        latencies = [it["metrics"].get("latency", 0) for it in self.iterations]
        efficiencies = [it["metrics"].get("token_efficiency", 0) for it in self.iterations]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Recursive Pattern Metrics: {self.name}", fontsize=16)

        axes[0, 0].bar(iterations, prompt_tokens, label="Prompt Tokens", color="blue", alpha=0.7)
        axes[0, 0].bar(iterations, response_tokens, bottom=prompt_tokens, 
                       label="Response Tokens", color="green", alpha=0.7)
        axes[0, 0].set_title("Token Usage by Iteration")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Tokens")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(iterations, latencies, marker='o', color="red", alpha=0.7)
        axes[0, 1].set_title("Latency by Iteration")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Seconds")
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(iterations, efficiencies, marker='s', color="purple", alpha=0.7)
        axes[1, 0].set_title("Token Efficiency (Response/Prompt)")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Ratio")
        axes[1, 0].grid(alpha=0.3)

        cumulative_tokens = np.cumsum([it["metrics"].get("total_tokens", 0) for it in self.iterations])
        axes[1, 1].plot(iterations, cumulative_tokens, marker='^', color="orange", alpha=0.7)
        axes[1, 1].set_title("Cumulative Token Usage")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Total Tokens")
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()


# --- SelfReflection using Pydantic model output ---

class SelfReflection(RecursivePattern):
    """
    A recursive pattern that implements self-reflection and
    continuous improvement through meta-cognitive processes.
    """

    def __init__(
        self,
        reflection_template: str = "Analyze your previous response:\n\n{previous_response}\n\nIdentify strengths and weaknesses. How can you improve your response to better address the original query:\n\n{original_query}",
        improvement_threshold: float = 0.8,
        **kwargs
    ):
        name = kwargs.pop("name", "Self-Reflection Pattern")
        description = kwargs.pop("description", "A pattern for continuous improvement through meta-cognitive processes")
        super().__init__(name=name, description=description, **kwargs)
        self.reflection_template = reflection_template
        self.improvement_threshold = improvement_threshold
        self.state["improvement_scores"] = []

    def _generate_recursive_prompt(self, iteration: int, **kwargs) -> str:
        input_query = kwargs.get("input")
        if iteration == 1:
            prompt = f"Please respond to the following query:\n\n{input_query}"
        else:
            previous_response_obj = kwargs.get("current_output", None)
            previous_response = previous_response_obj.response if hasattr(previous_response_obj, "response") else previous_response_obj
            prompt = self.reflection_template.format(
                previous_response=previous_response,
                original_query=input_query
            )
        return prompt

    def _process_response(self, response: Union[str, SelfReflectionResponse], iteration: int) -> SelfReflectionResponse:
        # Use the parsed model if available, else fallback to parsing response.
        if isinstance(response, SelfReflectionResponse):
            result = response
        else:
            # Heuristic parse
            score_pattern = r"(?:improvement|quality)\s*(?:score|rating)?:?\s*(\d+(?:\.\d+)?)\s*(?:\/\s*10)?"
            score_match = re.search(score_pattern, response.lower())
            improvement_score = float(score_match.group(1)) / 10 if score_match else 0.5
            result = SelfReflectionResponse(response=response, improvement_score=improvement_score)
        # Track score in state
        self.state.setdefault("improvement_scores", []).append(result.improvement_score)
        return result

    def _should_continue(self, iteration: int, current_output: SelfReflectionResponse) -> bool:
        if iteration >= self.max_iterations:
            return False
        if iteration == 1:
            return True
        if current_output.improvement_score >= self.improvement_threshold:
            self._log(f"Reached improvement threshold: {current_output.improvement_score:.2f}")
            return False
        return True

    def _call_llm(self, prompt: str, custom_system_message: Optional[str] = None) -> Tuple[SelfReflectionResponse, Dict[str, Any]]:
        return super(SelfReflection, self)._call_llm(
            prompt, custom_system_message, response_model=SelfReflectionResponse if self.iterations else None
        )

    def run(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        final_output, iterations = super().run(input_data)
        example_dir = OUTPUT_ROOT / "example_self_reflection"
        save_recursive_pattern(
            example_dir=example_dir,
            pattern_name=self.name,
            input_data=self.state.get("input"),
            iterations=self.iterations,
            final_output=self.state.get("last_output"),
            metrics=self.get_summary_metrics()
        )
        return final_output, self.iterations

# --- RecursiveBootstrapping using Pydantic model output ---

class RecursiveBootstrapping(RecursivePattern):
    """
    A recursive pattern that bootstraps its own capabilities
    by generating increasingly sophisticated strategies.
    """

    def __init__(
        self,
        bootstrap_template: str = "Based on your current approach to solving this problem:\n\n{current_approach}\n\nGenerate a more sophisticated strategy that builds upon your current approach and addresses its limitations.",
        sophistication_levels: List[str] = None,
        **kwargs
    ):
        name = kwargs.pop("name", "Recursive Bootstrapping Pattern")
        description = kwargs.pop("description", "A pattern for bootstrapping increasingly sophisticated strategies")
        super().__init__(name=name, description=description, **kwargs)
        self.bootstrap_template = bootstrap_template
        self.sophistication_levels = sophistication_levels if sophistication_levels is not None else [
            "basic", "intermediate", "advanced", "expert", "innovative"
        ]
        self.state["sophistication_level"] = 0

    def _generate_recursive_prompt(self, iteration: int, **kwargs) -> str:
        input_problem = kwargs.get("input")
        if iteration == 1:
            level = self.sophistication_levels[0]
            return f"""You are solving the following problem:\n{input_problem}\nStart by developing a {level} approach to solve this problem.\nFocus on foundational concepts and straightforward techniques."""
        current_approach = kwargs.get("current_output", {}).approach if hasattr(kwargs.get("current_output", {}), "approach") else kwargs.get("current_output", {}).get("approach", "")
        level_idx = min(iteration - 1, len(self.sophistication_levels) - 1)
        current_level = self.sophistication_levels[level_idx - 1]
        next_level = self.sophistication_levels[level_idx]
        return f"""You are solving the following problem:\n{input_problem}\nYour current {current_level} approach is:\n{current_approach}\nNow, bootstrap to a {next_level} approach that builds upon your current strategy and addresses its limitations.\nYour new approach should be more sophisticated, nuanced, and effective."""

    def _process_response(self, response: Union[str, BootstrappingResponse], iteration: int) -> BootstrappingResponse:
        if isinstance(response, BootstrappingResponse):
            result = response
        else:
            # Fallback parse: entire response as approach
            result = BootstrappingResponse(approach=str(response))
        # Track sophistication level
        level_idx = min(iteration - 1, len(self.sophistication_levels) - 1)
        self.state["sophistication_level"] = level_idx
        return result

    def _call_llm(self, prompt: str, custom_system_message: Optional[str] = None) -> Tuple[BootstrappingResponse, Dict[str, Any]]:
        return super(RecursiveBootstrapping, self)._call_llm(
            prompt, custom_system_message, response_model=BootstrappingResponse
        )

    def run(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        final_output, iterations = super().run(input_data)
        example_dir = OUTPUT_ROOT / "example_recursive_bootstrapping"
        save_recursive_pattern(
            example_dir=example_dir,
            pattern_name=self.name,
            input_data=self.state.get("input"),
            iterations=self.iterations,
            final_output=self.state.get("last_output"),
            metrics=self.get_summary_metrics()
        )
        return final_output, self.iterations

# --- SymbolicResidue using Pydantic model output ---

class SymbolicResidue(RecursivePattern):
    """
    A recursive pattern that tracks, integrates, and evolves
    symbolic residue across iterations.
    """

    def __init__(
        self,
        residue_template: str = "Process the following input while surfacing and integrating symbolic residue:\n\nInput: {input}\n\nCurrent symbolic residue: {symbolic_residue}",
        **kwargs
    ):
        name = kwargs.pop("name", "Symbolic Residue Pattern")
        description = kwargs.pop("description", "A pattern for tracking and integrating symbolic residue")
        super().__init__(name=name, description=description, **kwargs)
        self.residue_template = residue_template
        self.state["symbolic_residue"] = []
        self.state["residue_compression"] = 0.0
        self.state["resonance_score"] = 0.0

    def _generate_recursive_prompt(self, iteration: int, **kwargs) -> str:
        input_data = kwargs.get("input")
        residue_text = "\n".join([f"- {item}" for item in self.state.get("symbolic_residue", [])]) or "None yet"
        if iteration == 1:
            return f"""Process the following input and surface symbolic residue:\nInput: {input_data}\nSurface any symbolic residue or patterns (i.e., fragments or echoes emerging but not directly part of the output). Your response should include:\n1. The processed output\n2. A section titled "Surfaced Symbolic Residue" listing the residue\n3. A resonance score (0.0-1.0) indicating resonance with input."""
        return f"""Process the following input while integrating residue:\nInput: {input_data}\nCurrent symbolic residue:\n{residue_text}\nResidue compression: {self.state.get('residue_compression', 0.0):.2f}\nResonance score: {self.state.get('resonance_score', 0.0):.2f}\nIntegrate residue, then surface new or evolved residue. Your response should include:\n1. The processed output with integrated residue\n2. A section titled "Evolved Symbolic Residue"\n3. A residue compression score (0.0-1.0)\n4. A resonance score (0.0-1.0)."""

    def _process_response(self, response: Union[str, ResidueResponse], iteration: int) -> ResidueResponse:
        if isinstance(response, ResidueResponse):
            result = response
        else:
            # fallback parse for initial round
            main_output = response
            residue_items = []
            compression_score = 0.0
            resonance_score = 0.0
            result = ResidueResponse(
                output=main_output,
                symbolic_residue=residue_items,
                residue_compression=compression_score,
                resonance_score=resonance_score
            )
        # Always update state even if values are all zero/empty
        self.state["symbolic_residue"] = result.symbolic_residue
        self.state["residue_compression"] = result.residue_compression
        self.state["resonance_score"] = result.resonance_score
        return result

    def _call_llm(self, prompt: str, custom_system_message: Optional[str] = None) -> Tuple[ResidueResponse, Dict[str, Any]]:
        return super(SymbolicResidue, self)._call_llm(
            prompt, custom_system_message, response_model=ResidueResponse
        )

    def run(self, input_data: Any) -> Tuple[Any, List[Dict[str, Any]]]:
        final_output, iterations = super().run(input_data)
        example_dir = OUTPUT_ROOT / "example_symbolic_residue"
        save_recursive_pattern(
            example_dir=example_dir,
            pattern_name=self.name,
            input_data=self.state.get("input"),
            iterations=self.iterations,
            final_output=self.state.get("last_output"),
            metrics=self.get_summary_metrics()
        )
        return final_output, self.iterations

def example_self_reflection() -> Tuple[Any, List[Dict[str, Any]]]:
    input_question = (
        "How can AI language models be used effectively for structured context engineering in technical projects?"
    )
    pattern = SelfReflection(verbose=True, max_iterations=3)
    final_output, _ = pattern.run(input_question)
    return final_output, pattern.iterations

def example_recursive_bootstrapping() -> Tuple[Any, List[Dict[str, Any]]]:
    input_problem = "Design a workflow for iterative schema evolution in a knowledge graph."
    pattern = RecursiveBootstrapping(verbose=True, max_iterations=4)
    final_output, _ = pattern.run(input_problem)
    return final_output, pattern.iterations

def example_symbolic_residue() -> Tuple[Any, List[Dict[str, Any]]]:
    input_note = (
        "Document the recurring motifs in user interaction data collected from three different applications."
    )
    pattern = SymbolicResidue(verbose=True, max_iterations=3)
    final_output, _ = pattern.run(input_note)
    return final_output, pattern.iterations

if __name__ == "__main__":
    print("Running all recursive pattern examples and saving outputs...")

    example_self_reflection()
    example_recursive_bootstrapping()
    example_symbolic_residue()

    print(f"\nAll outputs saved under: {OUTPUT_ROOT}")
