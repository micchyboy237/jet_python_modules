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
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM
import shutil
import pathlib

OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

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

# Initialize our LLM interface
llm = SimpleLLM()

# ----- EXPERIMENT 1: THE ATOMIC PROMPT -----
print("\n----- EXPERIMENT 1: THE ATOMIC PROMPT -----")
print("Let's start with the most basic unit: a single instruction.")

atomic_prompt = "Write a short poem about programming."
tokens = llm.count_tokens(atomic_prompt)

print(f"\nAtomic Prompt: '{atomic_prompt}'")
print(f"Token Count: {tokens}")
print("\nGenerating response...")
response, metadata = llm.generate_response(atomic_prompt)
print(f"\nResponse:\n{response}")

# ----- EXPERIMENT 2: ADDING CONSTRAINTS -----
print("\n----- EXPERIMENT 2: ADDING CONSTRAINTS -----")
print("Now let's add constraints to our atomic prompt and observe the difference.")

# Let's create three versions with increasing constraints
prompts = [
    "Write a short poem about programming.",  # Original
    "Write a short poem about programming in 4 lines.",  # Added length constraint
    "Write a short haiku about programming using only simple words."  # Format and vocabulary constraints
]

# Measure tokens and generate responses
results = []
for i, prompt in enumerate(prompts):
    tokens = llm.count_tokens(prompt)
    print(f"\nPrompt {i+1}: '{prompt}'")
    print(f"Token Count: {tokens}")
    
    start_time = time.time()
    response, metadata = llm.generate_response(prompt)
    end_time = time.time()
    
    results.append({
        "prompt": prompt,
        "tokens": tokens,
        "response": response,
        "latency": end_time - start_time
    })
    
    print(f"Latency: {results[-1]['latency']:.4f} seconds")
    print(f"Response:\n{response}")

# ----- EXPERIMENT 3: MEASURING THE ROI CURVE -----
print("\n----- EXPERIMENT 3: MEASURING THE ROI CURVE -----")
print("Let's explore the relationship between prompt complexity and output quality.")

# In a real notebook, you would define subjective quality scores for each response
# For this demo, we'll use placeholder values
quality_scores = [3, 6, 8]  # Placeholder subjective scores on a scale of 1-10

# Plot tokens vs. quality
plt.figure(figsize=(10, 6))
tokens_list = [r["tokens"] for r in results]
plt.plot(tokens_list, quality_scores, marker='o', linestyle='-', color='blue')
plt.xlabel('Tokens in Prompt')
plt.ylabel('Output Quality (1-10)')
plt.title('Token-Quality ROI Curve')
plt.grid(True)

# Add annotations
for i, (x, y) in enumerate(zip(tokens_list, quality_scores)):
    plt.annotate(f"Prompt {i+1}", (x, y), textcoords="offset points", 
                 xytext=(0, 10), ha='center')

# Show the plot (in Jupyter this would display inline)
# plt.show()
print("[A plot would display here in a Jupyter environment]")

# ----- EXPERIMENT 4: MINIMAL CONTEXT ENHANCEMENT -----
print("\n----- EXPERIMENT 4: MINIMAL CONTEXT ENHANCEMENT -----")
print("Now we'll add minimal context to improve output quality while keeping token count low.")

# Let's create a prompt with a small amount of strategic context
enhanced_prompt = """Task: Write a haiku about programming.

A haiku is a three-line poem with 5, 7, and 5 syllables per line.
Focus on the feeling of solving a difficult bug."""

tokens = llm.count_tokens(enhanced_prompt)
print(f"\nEnhanced Prompt:\n'{enhanced_prompt}'")
print(f"Token Count: {tokens}")

response, metadata = llm.generate_response(enhanced_prompt)
print(f"\nResponse:\n{response}")

# ----- EXPERIMENT 5: MEASURING CONSISTENCY -----
print("\n----- EXPERIMENT 5: MEASURING CONSISTENCY -----")
print("Let's test how consistent the outputs are with minimal vs. enhanced prompts.")

# Function to generate multiple responses and measure consistency
def measure_consistency(prompt: str, n_samples: int = 3) -> Dict[str, Any]:
    """Generate multiple responses and measure consistency metrics."""
    responses = []
    total_tokens = 0
    
    for _ in range(n_samples):
        response, metadata = llm.generate_response(prompt)
        responses.append(response)
        total_tokens += llm.count_tokens(prompt)
    
    # In a real notebook, you would implement proper consistency metrics
    # such as semantic similarity between responses
    consistency_score = 0.5  # Placeholder value
    
    return {
        "prompt": prompt,
        "responses": responses,
        "total_tokens": total_tokens,
        "consistency_score": consistency_score
    }

# Compare basic vs enhanced prompt
basic_results = measure_consistency(prompts[0])
enhanced_results = measure_consistency(enhanced_prompt)

print(f"\nBasic Prompt Consistency Score: {basic_results['consistency_score']}")
print(f"Enhanced Prompt Consistency Score: {enhanced_results['consistency_score']}")

# ----- CONCLUSION -----
print("\n----- CONCLUSION -----")
print("Key insights from our experiments:")
print("1. Even small additions to prompts can significantly impact output quality")
print("2. There's an ROI curve where token count and quality find an optimal balance")
print("3. Adding minimal but strategic context improves consistency")
print("4. The best prompts are clear, concise, and provide just enough context")

print("\nTotal tokens used:", llm.get_stats()["total_tokens"])
print("\nToken efficiency:", llm.get_stats()["token_efficiency"])
print("\nTokens per second:", llm.get_stats()["tokens_per_second"])

# ----- NEXT STEPS -----
print("\n----- NEXT STEPS -----")
print("1. Try these experiments with a real LLM API")
print("2. Implement proper consistency and quality metrics")
print("3. Explore the concept of 'molecules' - combining multiple instructions")
print("4. Experiment with few-shot examples in the context window")

"""
EXERCISE FOR THE READER:

1. Connect this notebook to a real LLM API (OpenAI, Anthropic, etc.)
2. Test the same prompts with different model sizes
3. Create your own token-quality curve for a task you care about
4. Find the "minimum viable context" for your specific use case

See 02_expand_context.ipynb for more advanced context engineering techniques!
"""

# If this were a Jupyter notebook, we'd save the results to a file here
# with open('experiment_results.json', 'w') as f:
#     json.dump(results, f, indent=2)
