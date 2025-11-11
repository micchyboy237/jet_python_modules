#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context Expansion Techniques: From Prompts to Layered Context
=============================================================

This notebook presents hands-on strategies for evolving basic prompts into layered, information-rich contexts that enhance LLM performance. The focus is on practical context engineering: how to strategically add and structure context layers, and systematically measure the effects on both token usage and output quality.

Key concepts covered:
1. Transforming minimal prompts into expanded, context-rich structures
2. Principles of context layering and compositional prompt engineering
3. Quantitative measurement of token usage as context grows
4. Qualitative assessment of model output improvements
5. Iterative approaches to context refinement and optimization

Usage:
    # In Jupyter or Colab:
    %run 02_context_expansion.py
    # or
    # Step through notebook cells, modifying context layers and observing effects

Notes:
    - Each section is modular—experiment by editing and running different context layers.
    - Track how additional context alters both cost (token count) and performance (output quality).
    - Use as a practical foundation for developing advanced context engineering protocols.
"""

import os
import json
import time
import tiktoken  # OpenAI's tokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

# Load environment variables (you'll need to add your API key in a .env file)
import dotenv

dotenv.load_dotenv()

from jet._token.token_utils import token_counter
from jet.adapters.llama_cpp.llm import LlamacppLLM
import shutil
import pathlib

OUTPUT_DIR = pathlib.Path(__file__).parent / "generated" / pathlib.Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Create timestamped run directory under OUTPUT_DIR
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen3-instruct-2507:4b"
client = LlamacppLLM(model=MODEL, verbose=True)

def count_tokens(text: str) -> int:
    """Count tokens in a string using the appropriate tokenizer."""
    return token_counter(text, model=MODEL)

def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def calculate_metrics(prompt: str, response: str, latency: float) -> Dict[str, float]:
    """Calculate key metrics for a prompt-response pair."""
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response)
    token_efficiency = response_tokens / prompt_tokens if prompt_tokens > 0 else 0
    latency_per_1k = (latency / prompt_tokens) * 1000 if prompt_tokens > 0 else 0
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "token_efficiency": token_efficiency,
        "latency": latency,
        "latency_per_1k": latency_per_1k
    }

def generate_response(prompt: str) -> Tuple[str, float]:
    """Generate a response from the LLM and measure latency."""
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
    return response, latency

# Base prompt (atom)
base_prompt = "Write a paragraph about climate change."

# Expanded prompt variations (molecules)
expanded_prompts = {
    "base": base_prompt,
    "with_role": """You are an environmental scientist with expertise in climate systems. 
Write a paragraph about climate change.""",
    "with_examples": """Write a paragraph about climate change.

Example 1:
Climate change refers to long-term shifts in temperatures and weather patterns. Human activities have been the main driver of climate change since the 1800s, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.

Example 2:
Global climate change is evident in the increasing frequency of extreme weather events, rising sea levels, and shifting wildlife populations. Scientific consensus points to human activity as the primary cause.""",
    "with_constraints": """Write a paragraph about climate change.
- Include at least one scientific fact with numbers
- Mention both causes and effects
- End with a call to action
- Keep the tone informative but accessible""",
    "with_audience": """Write a paragraph about climate change for high school students who are
just beginning to learn about environmental science. Use clear explanations 
and relatable examples.""",
    "comprehensive": """You are an environmental scientist with expertise in climate systems.

Write a paragraph about climate change for high school students who are
just beginning to learn about environmental science. Use clear explanations 
and relatable examples.

Guidelines:
- Include at least one scientific fact with numbers
- Mention both causes and effects
- End with a call to action
- Keep the tone informative but accessible

Example of tone and structure:
"Ocean acidification occurs when seawater absorbs CO2 from the atmosphere, causing pH levels to drop. Since the Industrial Revolution, ocean pH has decreased by 0.1 units, representing a 30% increase in acidity. This affects marine life, particularly shellfish and coral reefs, as it impairs their ability to form shells and skeletons. Scientists predict that if emissions continue at current rates, ocean acidity could increase by 150% by 2100, devastating marine ecosystems. By reducing our carbon footprint through simple actions like using public transportation, we can help protect these vital ocean habitats."
"""
}

# Run experiments and save results
results = {}
responses = {}
for name, prompt in expanded_prompts.items():
    response, latency = generate_response(prompt)
    responses[name] = response
    metrics = calculate_metrics(prompt, response, latency)
    results[name] = metrics

    # Save individual prompt/response
    exp_dir = RUN_DIR / "01_prompt_variants" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (exp_dir / "response.txt").write_text(response, encoding="utf-8")
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Prepare data for visualization
prompt_types = list(results.keys())
prompt_tokens = [results[k]['prompt_tokens'] for k in prompt_types]
response_tokens = [results[k]['response_tokens'] for k in prompt_types]
latencies = [results[k]['latency'] for k in prompt_types]

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Token Usage
axes[0, 0].bar(prompt_types, prompt_tokens, label='Prompt Tokens', alpha=0.7, color='blue')
axes[0, 0].bar(prompt_types, response_tokens, bottom=prompt_tokens, label='Response Tokens', alpha=0.7, color='green')
axes[0, 0].set_title('Token Usage by Prompt Type')
axes[0, 0].set_ylabel('Number of Tokens')
axes[0, 0].legend()
plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

# Plot 2: Token Efficiency (Response Tokens / Prompt Tokens)
token_efficiency = [results[k]['token_efficiency'] for k in prompt_types]
axes[0, 1].bar(prompt_types, token_efficiency, color='purple', alpha=0.7)
axes[0, 1].set_title('Token Efficiency (Response/Prompt)')
axes[0, 1].set_ylabel('Efficiency Ratio')
plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

# Plot 3: Latency
axes[1, 0].bar(prompt_types, latencies, color='red', alpha=0.7)
axes[1, 0].set_title('Response Latency')
axes[1, 0].set_ylabel('Seconds')
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

# Plot 4: Latency per 1k tokens
latency_per_1k = [results[k]['latency_per_1k'] for k in prompt_types]
axes[1, 1].bar(prompt_types, latency_per_1k, color='orange', alpha=0.7)
axes[1, 1].set_title('Latency per 1k Tokens')
axes[1, 1].set_ylabel('Seconds per 1k Tokens')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plot_path = RUN_DIR / "01_prompt_variants" / "token_analysis.png"
fig.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Plot saved: {plot_path}")

# Save all responses in a single summary
summary_path = RUN_DIR / "01_prompt_variants" / "all_responses.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    for name, response in responses.items():
        f.write(f"=== {name.upper()} ===\n\n")
        f.write(response)
        f.write("\n\n" + "=" * 80 + "\n")
print(f"All responses saved: {summary_path}")

def create_expanded_context(
    base_prompt: str, 
    role: Optional[str] = None,
    examples: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    audience: Optional[str] = None,
    tone: Optional[str] = None,
    output_format: Optional[str] = None
) -> str:
    """
    Create an expanded context from a base prompt with optional components.
    """
    context_parts = []
    if role:
        context_parts.append(f"You are {role}.")
    context_parts.append(base_prompt)
    if audience:
        context_parts.append(f"Your response should be suitable for {audience}.")
    if tone:
        context_parts.append(f"Use a {tone} tone in your response.")
    if output_format:
        context_parts.append(f"Format your response as {output_format}.")
    if constraints and len(constraints) > 0:
        context_parts.append("Requirements:")
        for constraint in constraints:
            context_parts.append(f"- {constraint}")
    if examples and len(examples) > 0:
        context_parts.append("Examples:")
        for i, example in enumerate(examples, 1):
            context_parts.append(f"Example {i}:\n{example}")
    expanded_context = "\n\n".join(context_parts)
    return expanded_context

# Test our template
new_base_prompt = "Explain how photosynthesis works."
new_expanded_context = create_expanded_context(
    base_prompt=new_base_prompt,
    role="a biology teacher with 15 years of experience",
    audience="middle school students",
    tone="enthusiastic and educational",
    constraints=[
        "Use a plant-to-factory analogy",
        "Mention the role of chlorophyll",
        "Explain the importance for Earth's ecosystem",
        "Keep it under 200 words"
    ],
    examples=[
        "Photosynthesis is like a tiny factory inside plants. Just as a factory needs raw materials, energy, and workers to make products, plants need carbon dioxide, water, sunlight, and chlorophyll to make glucose (sugar) and oxygen. The sunlight is the energy source, chlorophyll molecules are the workers that capture this energy, while carbon dioxide and water are the raw materials. The factory's products are glucose, which the plant uses for growth and energy storage, and oxygen, which is released into the air for animals like us to breathe. This process is essential for life on Earth because it provides the oxygen we need and removes carbon dioxide from the atmosphere."
    ]
)

template_dir = RUN_DIR / "02_template_example"
template_dir.mkdir(parents=True, exist_ok=True)
(template_dir / "expanded_context.txt").write_text(new_expanded_context, encoding="utf-8")
print(f"Template context saved: {template_dir / 'expanded_context.txt'}")
print(f"Token count: {count_tokens(new_expanded_context)}")

# Generate a response using our expanded context
response, latency = generate_response(new_expanded_context)
metrics = calculate_metrics(new_expanded_context, response, latency)
(template_dir / "response.txt").write_text(response, encoding="utf-8")
(template_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(f"Response saved: {template_dir / 'response.txt'}")
print(f"Response tokens: {metrics['response_tokens']}")
print(f"Latency: {metrics['latency']:.2f}s")

def test_layered_contexts(base_prompt: str, context_layers: Dict[str, str]) -> Dict[str, Dict]:
    """
    Test different combinations of context layers to find optimal configurations.
    """
    layer_results = {}
    base_response, base_latency = generate_response(base_prompt)
    layer_results["base"] = {
        "prompt": base_prompt,
        "response": base_response,
        **calculate_metrics(base_prompt, base_response, base_latency)
    }
    for layer_name, layer_content in context_layers.items():
        combined_prompt = f"{base_prompt}\n\n{layer_content}"
        response, latency = generate_response(combined_prompt)
        layer_results[f"base+{layer_name}"] = {
            "prompt": combined_prompt,
            "response": response,
            **calculate_metrics(combined_prompt, response, latency)
        }
    all_layers = "\n\n".join(context_layers.values())
    full_prompt = f"{base_prompt}\n\n{all_layers}"
    full_response, full_latency = generate_response(full_prompt)
    layer_results["all_layers"] = {
        "prompt": full_prompt,
        "response": full_response,
        **calculate_metrics(full_prompt, full_response, full_latency)
    }
    return layer_results

# Define a base prompt and separate context layers
layer_test_prompt = "Write code to implement a simple weather app."
context_layers = {
    "role": "You are a senior software engineer with expertise in full-stack development and UI/UX design.",
    "requirements": """Requirements:
- The app should show current temperature, conditions, and forecast for the next 3 days
- It should allow users to search for weather by city name
- It should have a clean, responsive interface
- The app should handle error states gracefully""",
    "tech_stack": """Technical specifications:
- Use HTML, CSS, and vanilla JavaScript (no frameworks)
- Use the OpenWeatherMap API for weather data
- All code should be well-commented and follow best practices
- Include both the HTML structure and JavaScript functionality""",
    "example": """Example structure (but improve upon this):
```html
<!DOCTYPE html>
<html>
<head>
    <title>Weather App</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Weather App</h1>
        <div class="search">
            <input type="text" placeholder="Enter city name">
            <button>Search</button>
        </div>
        <div class="weather-display">
            <!-- Weather data will be displayed here -->
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
```"""
}

# Run the layer optimization test
layer_test_results = test_layered_contexts(layer_test_prompt, context_layers)

# Extract data for visualization
config_names = list(layer_test_results.keys())
prompt_sizes = [layer_test_results[k]['prompt_tokens'] for k in config_names]
response_sizes = [layer_test_results[k]['response_tokens'] for k in config_names]
efficiencies = [layer_test_results[k]['token_efficiency'] for k in config_names]

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].bar(config_names, prompt_sizes, label='Prompt Tokens', alpha=0.7, color='blue')
axes[0].bar(config_names, response_sizes, label='Response Tokens', alpha=0.7, color='green')
axes[0].set_title('Token Usage by Context Configuration')
axes[0].set_ylabel('Number of Tokens')
axes[0].legend()
plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right')

axes[1].bar(config_names, efficiencies, color='purple', alpha=0.7)
axes[1].set_title('Token Efficiency by Context Configuration')
axes[1].set_ylabel('Efficiency Ratio (Response/Prompt)')
plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
layer_plot_path = RUN_DIR / "03_layered_contexts" / "layer_analysis.png"
layer_plot_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(layer_plot_path, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Layer analysis plot saved: {layer_plot_path}")

# Identify the most efficient configuration and save results
most_efficient = max(config_names, key=lambda x: layer_test_results[x]['token_efficiency'])
layer_summary = RUN_DIR / "03_layered_contexts" / "summary.txt"
with open(layer_summary, "w", encoding="utf-8") as f:
    f.write(f"Most token-efficient: {most_efficient}\n")
    f.write(f"Efficiency ratio: {layer_test_results[most_efficient]['token_efficiency']:.2f}\n\n")
    f.write("All configurations:\n")
    for name in config_names:
        data = layer_test_results[name]
        f.write(f"- {name}: {data['token_efficiency']:.3f} efficiency, {data['prompt_tokens']} prompt tokens\n")
print(f"Layer summary saved: {layer_summary}")

with open(RUN_DIR / "03_layered_contexts" / "full_results.json", "w", encoding="utf-8") as f:
    json.dump(layer_test_results, f, indent=2, ensure_ascii=False)

def compress_context(context: str, technique: str = 'summarize') -> str:
    """
    Apply different compression techniques to reduce token usage while preserving meaning.
    """
    if technique == 'summarize':
        prompt = f"""Summarize the following context in a concise way that preserves all key information
but uses fewer words. Focus on essential instructions and details:

{context}"""
        compressed, _ = generate_response(prompt)
        return compressed
    elif technique == 'keywords':
        prompt = f"""Extract the most important keywords, phrases, and instructions from this context:

{context}

Format your response as a comma-separated list of essential terms and short phrases."""
        keywords, _ = generate_response(prompt)
        return keywords
    elif technique == 'bullet':
        prompt = f"""Convert this context into a concise, structured list of bullet points that
captures all essential information with minimal words:

{context}"""
        bullets, _ = generate_response(prompt)
        return bullets
    else:
        return context  # No compression

# Test compression on our comprehensive example
original_context = expanded_prompts["comprehensive"]
comp_dir = RUN_DIR / "04_compression_techniques"
comp_dir.mkdir(parents=True, exist_ok=True)
original_tokens = count_tokens(original_context)
(comp_dir / "original_context.txt").write_text(original_context, encoding="utf-8")

for technique in ['summarize', 'keywords', 'bullet']:
    compressed = compress_context(original_context, technique)
    compressed_tokens = count_tokens(compressed)
    ratio = compressed_tokens / original_tokens

    tech_dir = comp_dir / technique
    tech_dir.mkdir(exist_ok=True)
    (tech_dir / "compressed.txt").write_text(compressed, encoding="utf-8")
    (tech_dir / "stats.txt").write_text(
        f"Original tokens: {original_tokens}\n"
        f"Compressed tokens: {compressed_tokens}\n"
        f"Compression ratio: {ratio:.3f}\n",
        encoding="utf-8"
    )
    print(f"{technique.title()} compression saved: {tech_dir}")

def evaluate_response_quality(prompt: str, response: str, criteria: List[str]) -> float:
    """
    Use the LLM to evaluate the quality of a response based on specific criteria.
    """
    criteria_list = "\n".join([f"- {c}" for c in criteria])
    eval_prompt = f"""Rate the quality of the following response to a prompt. 
    
Prompt: 
{prompt}

Response:
{response}

Please evaluate based on these criteria:
{criteria_list}

For each criterion, rate from 0-10, then provide an overall score from 0.0 to 1.0 where 
1.0 is perfect and 0.0 is completely inadequate. Format your response as:

Criterion 1: [score] - [brief comment]
Criterion 2: [score] - [brief comment]
...
Overall Score: [0.0-1.0]
"""
    evaluation, _ = generate_response(eval_prompt)
    try:
        import re
        score_match = re.findall(r"Overall Score:\s*([0-9]*\.?[0-9]+)", evaluation)
        if score_match:
            return float(score_match[-1])
        else:
            return 0.5
    except:
        return 0.5

def prune_context_layers(base_prompt: str, layers: Dict[str, str], criteria: List[str]) -> Tuple[str, Dict]:
    """
    Systematically test and prune context layers that don't improve response quality.
    """
    print("Testing baseline...")
    base_response, base_latency = generate_response(base_prompt)
    base_quality = evaluate_response_quality(base_prompt, base_response, criteria)

    results = {
        "base": {
            "prompt": base_prompt,
            "response": base_response,
            "quality": base_quality,
            "tokens": count_tokens(base_prompt),
            "latency": base_latency
        }
    }

    all_layers_text = "\n\n".join(layers.values())
    full_prompt = f"{base_prompt}\n\n{all_layers_text}"
    print("Testing all layers...")
    full_response, full_latency = generate_response(full_prompt)
    full_quality = evaluate_response_quality(full_prompt, full_response, criteria)

    results["all_layers"] = {
        "prompt": full_prompt,
        "response": full_response,
        "quality": full_quality,
        "tokens": count_tokens(full_prompt),
        "latency": full_latency
    }

    best_quality = full_quality
    best_config = "all_layers"

    for layer_to_remove in layers.keys():
        remaining_layers = {k: v for k, v in layers.items() if k != layer_to_remove}
        remaining_text = "\n\n".join(remaining_layers.values())
        test_prompt = f"{base_prompt}\n\n{remaining_text}"

        print(f"Testing without '{layer_to_remove}'...")
        test_response, test_latency = generate_response(test_prompt)
        test_quality = evaluate_response_quality(test_prompt, test_response, criteria)

        config_name = f"without_{layer_to_remove}"
        results[config_name] = {
            "prompt": test_prompt,
            "response": test_response,
            "quality": test_quality,
            "tokens": count_tokens(test_prompt),
            "latency": test_latency
        }

        if test_quality >= best_quality:
            best_quality = test_quality
            best_config = config_name

    if best_config == "all_layers":
        return full_prompt, results

    if best_config.startswith("without_"):
        removed_layer = best_config.replace("without_", "")
        remaining_layers = {k: v for k, v in layers.items() if k != removed_layer}
        print(f"Layer '{removed_layer}' can be removed. Testing further pruning...")
        return prune_context_layers(base_prompt, remaining_layers, criteria)

    return results[best_config]["prompt"], results

# Test context pruning (leave commented if not needed)
pruning_test_prompt = "Write a tutorial on how to use pandas for data analysis."
pruning_layers = {
    "role": "You are a data science instructor with 10+ years of experience teaching Python libraries.",
    "audience": "Your audience consists of beginner Python programmers who understand basic programming concepts but have no prior experience with data analysis.",
    "structure": "Structure the tutorial with these sections: Introduction, Installation, Loading Data, Basic Operations, Data Cleaning, Data Visualization, and a Practical Example.",
    "style": "Use a friendly, conversational tone. Include code snippets with comments explaining each line. Break down complex concepts into simple explanations.",
    "unnecessary": "Include details about the history of pandas and its development team. Mention that pandas was created by Wes McKinney in 2008 while he was at AQR Capital Management."
}
evaluation_criteria = [
    "Completeness - covers all essential concepts",
    "Clarity - explains concepts in an easy-to-understand way",
    "Code quality - provides useful, correct code examples",
    "Beginner-friendliness - assumes no prior knowledge of pandas",
    "Practicality - includes real-world applications"
]
# Uncomment to run the pruning test (takes time to run)
# optimized_prompt, pruning_results = prune_context_layers(pruning_test_prompt, pruning_layers, evaluation_criteria)
# print("\nOPTIMIZED PROMPT:")
# print("-" * 80)
# print(optimized_prompt)
# print("-" * 80)
# for config, data in pruning_results.items():
#     print(f"{config}: Quality = {data['quality']:.2f}, Tokens = {data['tokens']}")

def retrieve_relevant_info(query: str, knowledge_base: List[Dict[str, str]]) -> List[str]:
    """
    Retrieve relevant information from a knowledge base based on a query.
    """
    relevant_info = []
    query_terms = set(query.lower().split())
    for item in knowledge_base:
        content = item['content'].lower()
        title = item['title'].lower()
        matches = sum(1 for term in query_terms if term in content or term in title)
        if matches > 0:
            relevant_info.append(item['content'])
    return relevant_info[:3]

# Example knowledge base (in a real application, this would be much larger)
sample_knowledge_base = [
    {
        "title": "Pandas Introduction",
        "content": "Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. Key features include DataFrame objects, handling of missing data, and data alignment."
    },
    {
        "title": "Pandas Installation",
        "content": "To install pandas, run: pip install pandas. For Anaconda users, pandas comes pre-installed. You can import pandas with: import pandas as pd"
    },
    {
        "title": "Loading Data in Pandas",
        "content": "Pandas can read data from various sources including CSV, Excel, SQL databases, and JSON. Example: df = pd.read_csv('data.csv')"
    },
    {
        "title": "Data Cleaning with Pandas",
        "content": "Pandas provides functions for handling missing data, such as dropna() and fillna(). It also offers methods for removing duplicates and transforming data."
    },
    {
        "title": "Data Visualization with Pandas",
        "content": "Pandas integrates with matplotlib to provide plotting capabilities. Simple plots can be created with df.plot(). For more complex visualizations, use: import matplotlib.pyplot as plt"
    }
]

def create_rag_context(base_prompt: str, query: str, knowledge_base: List[Dict[str, str]]) -> str:
    """
    Create a retrieval-augmented context by combining a base prompt with relevant information.
    """
    relevant_info = retrieve_relevant_info(query, knowledge_base)
    if not relevant_info:
        return base_prompt
    context_block = "Relevant information:\n\n" + "\n\n".join(relevant_info)
    rag_context = f"{base_prompt}\n\n{context_block}"
    return rag_context

# Test retrieval-augmented context expansion
rag_test_prompt = "Write a brief tutorial on how to load data in pandas and handle missing values."
rag_context = create_rag_context(rag_test_prompt, "pandas loading data cleaning", sample_knowledge_base)

rag_dir = RUN_DIR / "05_rag_example"
rag_dir.mkdir(parents=True, exist_ok=True)
(rag_dir / "rag_context.txt").write_text(rag_context, encoding="utf-8")
print(f"RAG context saved: {rag_dir / 'rag_context.txt'}")
print(f"Token count: {count_tokens(rag_context)}")

# Generate response with RAG context
rag_response, rag_latency = generate_response(rag_context)
(rag_dir / "response.txt").write_text(rag_response, encoding="utf-8")
print(f"RAG response saved: {rag_dir / 'response.txt'}")

# Save knowledge base
with open(rag_dir / "knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(sample_knowledge_base, f, indent=2, ensure_ascii=False)

# Save final run summary
summary_path = RUN_DIR / "RUN_SUMMARY.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Context Expansion Experiments\n")
    f.write(f"Run ID: {RUN_ID}\n")
    f.write(f"Model: {MODEL}\n")
    f.write(f"Total experiments: 5\n\n")
    f.write("Sections:\n")
    f.write("  01_prompt_variants/    - Base vs expanded prompts\n")
    f.write("  02_template_example/   - Dynamic context builder\n")
    f.write("  03_layered_contexts/   - Layer combination analysis\n")
    f.write("  04_compression_techniques/ - Summarize/keywords/bullet\n")
    f.write("  05_rag_example/        - Retrieval-augmented generation\n")
print(f"\nAll results saved to: {RUN_DIR}")
print(f"Summary: {summary_path}")

if __name__ == "__main__":
    print(f"Starting context expansion experiments...")
    print(f"Results will be saved to: {RUN_DIR}\n")
    # The main experiment flow is already at module level
    # This block ensures it runs when executed as script
    pass