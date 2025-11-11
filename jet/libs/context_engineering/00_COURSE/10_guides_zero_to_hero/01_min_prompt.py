#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Prompt Exploration: Fundamentals of Context Engineering
==============================================================

This notebook introduces the core principles of context engineering by exploring minimal, atomic prompts and their direct impact on LLM output and behavior.

Key concepts covered:
1. Constructing atomic prompts for maximum clarity and control
2. Measuring effectiveness through token count and model response quality
3. Iterative prompt modification for rapid feedback cycles
4. Observing context drift and minimal prompt boundaries
5. Foundations for scaling from atomic prompts to protocolized shells

Usage:
    # In Jupyter or Colab:
    %run 01_min_prompt.py
    # or
    # Edit and run each section independently to experiment with prompt effects

Notes:
    - Each section of this notebook is designed for hands-on experimentation.
    - Modify prompts and observe changes in tokenization and output fidelity.
    - Use this as a foundation for building up to advanced context engineering workflows.

"""

import time
import logging
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import shutil
import pathlib
import json

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM

OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_MODEL = "qwen3-instruct-2507:4b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

# If you're using OpenAI's API, uncomment these lines and add your API key
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key as an environment variable

# If you're using another provider, adjust accordingly
# Dummy LLM class for demonstration purposes
class SimpleLLM:
    """Minimal LLM interface for demonstration."""

    def __init__(self, model_name: str = "dummy-model"):
        """Initialize LLM interface."""
        self.model_name = model_name
        self.metadata = {}

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using a very simple approximation.
        In production, use the tokenizer specific to your model.
        """
        # This is an extremely rough approximation, use a proper tokenizer in practice
        return len(text.split())

    def generate_response(
        self,
        prompt: str,
        # client=None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        # system_message: str = "You are a helpful assistant."
    ) -> Tuple[str, Dict[str, Any]]:
        client = LlamacppLLM(model=model, verbose=True)

        prompt_tokens = token_counter(prompt, model=model)
        system_tokens = token_counter(prompt, model=model)

        metadata = {
            "prompt_tokens": prompt_tokens,
            "system_tokens": system_tokens,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timestamp": time.time()
        }

        try:
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
            response_tokens = token_counter(response_text, model=model)
            metadata.update({
                "latency": latency,
                "response_tokens": response_tokens,
                "total_tokens": prompt_tokens + system_tokens + response_tokens,
                "token_efficiency": response_tokens / (prompt_tokens + system_tokens) if (prompt_tokens + system_tokens) > 0 else 0,
                "tokens_per_second": response_tokens / latency if latency > 0 else 0
            })

            self.metadata = metadata

            return response_text, metadata

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            metadata["error"] = str(e)
            return f"ERROR: {str(e)}", metadata

    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics."""
        return self.metadata

llm = SimpleLLM()

def save_example_artifacts(example_dir: pathlib.Path, data: Dict[str, Any]):
    """Save all artifacts for an example in its directory."""
    example_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt
    if "prompt" in data:
        (example_dir / "prompt.md").write_text(data["prompt"])

    # Save response
    if "response" in data:
        (example_dir / "response.md").write_text(data["response"])

    # Save metadata
    if "metadata" in data:
        (example_dir / "metadata.json").write_text(json.dumps(data["metadata"], indent=2))

    # Save summary
    summary_lines = []
    if "title" in data:
        summary_lines.append(f"# {data['title']}\n")
    if "description" in data:
        summary_lines.append(data["description"] + "\n")
    if "results" in data:
        summary_lines.append("## Results\n")
        for r in data["results"]:
            summary_lines.append(f"- Prompt tokens: {r.get('tokens', 'N/A')}")
            summary_lines.append(f"  Latency: {r.get('latency', 0):.4f}s")
    (example_dir / "SUMMARY.md").write_text("\n".join(summary_lines))

    # Save plot if provided
    if "plot_path" in data:
        plt.savefig(data["plot_path"])
        plt.close()

def example_1_atomic_prompt():
    """Experiment 1: Single atomic prompt."""
    example_dir = OUTPUT_DIR / "example_1_atomic_prompt"
    prompt = "Write a short poem about programming."
    tokens = llm.count_tokens(prompt)
    response, metadata = llm.generate_response(prompt)

    save_example_artifacts(example_dir, {
        "title": "Experiment 1: The Atomic Prompt",
        "description": "Testing the most basic instruction with no constraints.",
        "prompt": prompt,
        "response": response,
        "metadata": metadata
    })

def example_2_constraints():
    """Experiment 2: Adding constraints to atomic prompt."""
    example_dir = OUTPUT_DIR / "example_2_constraints"
    prompts = [
        "Write a short poem about programming.",
        "Write a short poem about programming in 4 lines.",
        "Write a short haiku about programming using only simple words."
    ]
    results = []
    for i, prompt in enumerate(prompts):
        tokens = llm.count_tokens(prompt)
        start_time = time.time()
        response, metadata = llm.generate_response(prompt)
        latency = time.time() - start_time
        results.append({
            "prompt": prompt,
            "tokens": tokens,
            "response": response,
            "latency": latency,
            "metadata": metadata
        })
        # Save individual response
        sub_dir = example_dir / f"variant_{i+1}"
        save_example_artifacts(sub_dir, {
            "title": f"Variant {i+1}",
            "prompt": prompt,
            "response": response,
            "metadata": metadata
        })

    # Save comparison summary
    save_example_artifacts(example_dir, {
        "title": "Experiment 2: Adding Constraints",
        "description": "Compare how constraints affect output length and style.",
        "results": results
    })

def example_3_roi_curve():
    """Experiment 3: Token-Quality ROI Curve."""
    example_dir = OUTPUT_DIR / "example_3_roi_curve"
    # Reuse results from example_2
    prompts = [
        "Write a short poem about programming.",
        "Write a short poem about programming in 4 lines.",
        "Write a short haiku about programming using only simple words."
    ]
    tokens_list = []
    quality_scores = [3, 6, 8]  # Subjective quality
    results = []

    for i, prompt in enumerate(prompts):
        tokens = llm.count_tokens(prompt)
        response, metadata = llm.generate_response(prompt)
        tokens_list.append(tokens)
        results.append({"tokens": tokens, "quality": quality_scores[i]})

    plt.figure(figsize=(10, 6))
    plt.plot(tokens_list, quality_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel('Tokens in Prompt')
    plt.ylabel('Output Quality (1-10)')
    plt.title('Token-Quality ROI Curve')
    plt.grid(True)
    for i, (x, y) in enumerate(zip(tokens_list, quality_scores)):
        plt.annotate(f"Prompt {i+1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

    plot_path = example_dir / "roi_curve.png"
    save_example_artifacts(example_dir, {
        "title": "Experiment 3: ROI Curve",
        "description": "Visualizing trade-off between prompt length and output quality.",
        "plot_path": str(plot_path),
        "results": results
    })

def example_4_context_enhancement():
    """Experiment 4: Minimal context enhancement."""
    example_dir = OUTPUT_DIR / "example_4_context_enhancement"
    enhanced_prompt = """Task: Write a haiku about programming.

A haiku is a three-line poem with 5, 7, and 5 syllables per line.
Focus on the feeling of solving a difficult bug."""
    tokens = llm.count_tokens(enhanced_prompt)
    response, metadata = llm.generate_response(enhanced_prompt)

    save_example_artifacts(example_dir, {
        "title": "Experiment 4: Minimal Context Enhancement",
        "description": "Adding structure and focus with minimal tokens.",
        "prompt": enhanced_prompt,
        "response": response,
        "metadata": metadata
    })

def example_5_consistency():
    """Experiment 5: Measuring output consistency."""
    example_dir = OUTPUT_DIR / "example_5_consistency"

    def measure_consistency(prompt: str, n_samples: int = 3, output_dir: pathlib.Path) -> List[Dict]:
        responses = []
        for i in range(n_samples):
            response, metadata = llm.generate_response(prompt)
            sub_dir = output_dir / f"sample_{i+1}"
            save_example_artifacts(sub_dir, {
                "title": f"Sample {i+1}",
                "prompt": prompt,
                "response": response,
                "metadata": metadata
            })
            responses.append(response)
        return responses

    basic_prompt = "Write a short poem about programming."
    enhanced_prompt = """Task: Write a haiku about programming.

A haiku is a three-line poem with 5, 7, and 5 syllables per line.
Focus on the feeling of solving a difficult bug."""

    basic_responses = measure_consistency(basic_prompt, n_samples=3, output_dir= example_dir / "basic")
    enhanced_responses = measure_consistency(enhanced_prompt, n_samples=3, output_dir= example_dir / "enhanced")

    consistency_score = 0.5  # Placeholder

    save_example_artifacts(example_dir, {
        "title": "Experiment 5: Consistency Comparison",
        "description": "Basic vs enhanced prompt consistency across 3 runs.",
        "results": [
            {"type": "basic", "consistency_score": consistency_score},
            {"type": "enhanced", "consistency_score": consistency_score}
        ]
    })

if __name__ == "__main__":
    print(f"Saving all experiment results to: {OUTPUT_DIR}")
    example_1_atomic_prompt()
    example_2_constraints()
    example_3_roi_curve()
    example_4_context_enhancement()
    example_5_consistency()
    print("All examples completed and saved.")
