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

import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def save_example_artifacts(example_dir: pathlib.Path, data: Dict[str, Any]):
    """Save all artifacts for an example."""
    example_dir.mkdir(parents=True, exist_ok=True)

    # Save prompt
    if "prompt" in data:
        (example_dir / "prompt.md").write_text(data["prompt"], encoding="utf-8")

    # Save response
    if "response" in data:
        (example_dir / "response.md").write_text(data["response"], encoding="utf-8")

    # Save metrics
    if "metrics" in data:
        (example_dir / "metrics.json").write_text(json.dumps(data["metrics"], indent=2), encoding="utf-8")

    # Save plot
    if "plot_path" in data:
        plt.savefig(data["plot_path"], bbox_inches="tight", dpi=150)
        plt.close()

    # Save extra files
    for key, content in data.get("extra_files", {}).items():
        path = example_dir / key
        if isinstance(content, dict):
            path.write_text(json.dumps(content, indent=2), encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

    # Save summary
    summary_lines = []
    if "title" in data:
        summary_lines.append(f"# {data['title']}\n")
    if "description" in data:
        summary_lines.append(data["description"] + "\n")
    if "results" in data:
        summary_lines.append("## Results\n")
        for r in data["results"]:
            summary_lines.append(f"- {r.get('name', 'Variant')}: {r.get('prompt_tokens', 0)} prompt tokens")
    (example_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

def example_1_prompt_variants():
    """Experiment 1: Base vs expanded prompts."""
    example_dir = OUTPUT_DIR / "example_1_prompt_variants"
    base_prompt = "Write a paragraph about climate change."
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

    results = {}
    responses = {}
    for name, prompt in expanded_prompts.items():
        response, latency = generate_response(prompt)
        responses[name] = response
        metrics = calculate_metrics(prompt, response, latency)
        results[name] = metrics

    # Save individual variants
    for name in expanded_prompts:
        sub_dir = example_dir / name
        save_example_artifacts(sub_dir, {
            "prompt": expanded_prompts[name],
            "response": responses[name],
            "metrics": results[name]
        })

    # Generate and save plot
    prompt_types = list(results.keys())
    prompt_tokens = [results[k]['prompt_tokens'] for k in prompt_types]
    response_tokens = [results[k]['response_tokens'] for k in prompt_types]
    token_efficiency = [results[k]['token_efficiency'] for k in prompt_types]
    latencies = [results[k]['latency'] for k in prompt_types]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(prompt_types, prompt_tokens, label='Prompt Tokens', alpha=0.7, color='blue')
    axes[0, 0].bar(prompt_types, response_tokens, bottom=prompt_tokens, label='Response Tokens', alpha=0.7, color='green')
    axes[0, 0].set_title('Token Usage by Prompt Type')
    axes[0, 0].set_ylabel('Number of Tokens')
    axes[0, 0].legend()
    plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

    axes[0, 1].bar(prompt_types, token_efficiency, color='purple', alpha=0.7)
    axes[0, 1].set_title('Token Efficiency (Response/Prompt)')
    axes[0, 1].set_ylabel('Efficiency Ratio')
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

    axes[1, 0].bar(prompt_types, latencies, color='red', alpha=0.7)
    axes[1, 0].set_title('Response Latency')
    axes[1, 0].set_ylabel('Seconds')
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

    latency_per_1k = [results[k]['latency_per_1k'] for k in prompt_types]
    axes[1, 1].bar(prompt_types, latency_per_1k, color='orange', alpha=0.7)
    axes[1, 1].set_title('Latency per 1k Tokens')
    axes[1, 1].set_ylabel('Seconds per 1k Tokens')
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plot_path = example_dir / "token_analysis.png"

    save_example_artifacts(example_dir, {
        "title": "Experiment 1: Prompt Variants",
        "description": "Comparing base prompt with role, examples, constraints, audience, and comprehensive versions.",
        "plot_path": str(plot_path),
        "results": [{"name": k, **v} for k, v in results.items()],
        "extra_files": {
            "all_responses.txt": "\n\n".join([f"=== {k.upper()} ===\n\n{v}" for k, v in responses.items()])
        }
    })

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

def example_2_template_builder():
    """Experiment 2: Dynamic context builder."""
    example_dir = OUTPUT_DIR / "example_2_template_builder"

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

    response, latency = generate_response(new_expanded_context)
    metrics = calculate_metrics(new_expanded_context, response, latency)

    save_example_artifacts(example_dir, {
        "title": "Experiment 2: Template Context Builder",
        "description": "Dynamically constructing context using role, audience, tone, constraints, and examples.",
        "prompt": new_expanded_context,
        "response": response,
        "metrics": metrics
    })

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

def example_3_layered_contexts():
    """Experiment 3: Layer combination analysis."""
    example_dir = OUTPUT_DIR / "example_3_layered_contexts"

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

    layer_test_results = test_layered_contexts(layer_test_prompt, context_layers)

    # Save individual configs
    for name, data in layer_test_results.items():
        sub_dir = example_dir / name
        save_example_artifacts(sub_dir, {
            "prompt": data["prompt"],
            "response": data["response"],
            "metrics": {k: v for k, v in data.items() if k not in ["prompt", "response"]}
        })

    # Generate plot
    config_names = list(layer_test_results.keys())
    prompt_sizes = [layer_test_results[k]['prompt_tokens'] for k in config_names]
    response_sizes = [layer_test_results[k]['response_tokens'] for k in config_names]
    efficiencies = [layer_test_results[k]['token_efficiency'] for k in config_names]

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
    plot_path = example_dir / "layer_analysis.png"

    most_efficient = max(config_names, key=lambda x: layer_test_results[x]['token_efficiency'])

    save_example_artifacts(example_dir, {
        "title": "Experiment 3: Layered Contexts",
        "description": "Testing combinations of role, requirements, tech stack, and examples.",
        "plot_path": str(plot_path),
        "results": [{"name": k, **v} for k, v in layer_test_results.items()],
        "extra_files": {
            "summary.txt": f"Most token-efficient: {most_efficient}\n"
                          f"Efficiency ratio: {layer_test_results[most_efficient]['token_efficiency']:.2f}\n\n"
                          "All configurations:\n" + "\n".join([
                              f"- {k}: {v['token_efficiency']:.3f} efficiency, {v['prompt_tokens']} prompt tokens"
                              for k, v in layer_test_results.items()
                          ]),
            "full_results.json": layer_test_results
        }
    })

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

def example_4_compression_techniques():
    """Experiment 4: Context compression."""
    example_dir = OUTPUT_DIR / "example_4_compression_techniques"
    # Fix: expanded_prompts is defined in example_1 — make local copy
    original_context = """You are an environmental scientist with expertise in climate systems.
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
    original_tokens = count_tokens(original_context)

    for technique in ['summarize', 'keywords', 'bullet']:
        compressed = compress_context(original_context, technique)
        compressed_tokens = count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0

        tech_dir = example_dir / technique
        save_example_artifacts(tech_dir, {
            "prompt": original_context,
            "response": compressed,
            "extra_files": {
                "stats.txt": f"Original tokens: {original_tokens}\n"
                            f"Compressed tokens: {compressed_tokens}\n"
                            f"Compression ratio: {ratio:.3f}\n",
                "original_context.txt": original_context  # ← Save here instead
            }
        })

    save_example_artifacts(example_dir, {
        "title": "Experiment 4: Compression Techniques",
        "description": "Comparing summarization, keyword extraction, and bullet formatting."
    })

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

def example_5_rag_example():
    """Experiment 5: Retrieval-Augmented Generation."""
    example_dir = OUTPUT_DIR / "example_5_rag_example"

    rag_test_prompt = "Write a brief tutorial on how to load data in pandas and handle missing values."
    rag_context = create_rag_context(rag_test_prompt, "pandas loading data cleaning", sample_knowledge_base)
    response, latency = generate_response(rag_context)

    save_example_artifacts(example_dir, {
        "title": "Experiment 5: RAG Example",
        "description": "Using retrieved knowledge to enhance prompt.",
        "prompt": rag_context,
        "response": response,
        "metrics": calculate_metrics(rag_context, response, latency),
        "extra_files": {
            "knowledge_base.json": json.dumps(sample_knowledge_base, indent=2, ensure_ascii=False)
        }
    })

def generate_run_summary():
    """Generate final RUN_SUMMARY.md."""
    summary_path = OUTPUT_DIR / "RUN_SUMMARY.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"""# Context Expansion Experiments Summary

**Run Time**: {now}
**Model**: {MODEL}
**Total Examples**: 5

## Experiments

- `example_1_prompt_variants/` - Base vs expanded prompts
- `example_2_template_builder/` - Dynamic context assembly
- `example_3_layered_contexts/` - Layer combination analysis
- `example_4_compression_techniques/` - Summarize/keywords/bullet
- `example_5_rag_example/` - Retrieval-augmented generation

All outputs saved in: `{OUTPUT_DIR}`
"""
    summary_path.write_text(summary, encoding="utf-8")

if __name__ == "__main__":
    print("Starting context expansion experiments...")
    print(f"Results will be saved to: {OUTPUT_DIR}\n")

    example_1_prompt_variants()
    example_2_template_builder()
    example_3_layered_contexts()
    example_4_compression_techniques()
    example_5_rag_example()
    generate_run_summary()

    print("All examples completed and saved.")