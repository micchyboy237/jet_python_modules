#!/usr/bin/env python3
"""
Multi-Agent Markdown Summarizer for llama.cpp (10k token limit)
Saves this as: markdown_summarizer.py
"""

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ChunkSummary:
    """Represents a summary of a text chunk"""

    chunk_id: str
    content_summary: str
    source_file: str
    chunk_index: int
    token_estimate: int


@dataclass
class FileSummary:
    """Represents a summary of an entire file"""

    file_path: str
    summaries: List[ChunkSummary] = field(default_factory=list)
    combined_summary: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class DirectorySummary:
    """Represents a summary of a directory"""

    dir_path: str
    file_summaries: List[FileSummary] = field(default_factory=list)
    combined_summary: str = ""


@dataclass
class AgentConfig:
    """Configuration for the multi-agent system"""

    max_tokens_per_chunk: int = 8000
    max_output_tokens: int = 1500
    validation_threshold: float = 0.8
    max_validation_iterations: int = 2
    llama_server_url: str = "http://localhost:8080"
    temperature: float = 0.3
    batch_size: int = 3


class TokenEstimator:
    """Estimate tokens in text (rough approximation)"""

    @staticmethod
    def estimate(text: str) -> int:
        return len(text) // 4 + len(text.split()) // 2

    @staticmethod
    def safe_chunk_size(max_tokens: int, safety_margin: float = 0.7) -> int:
        return int(max_tokens * 4 * safety_margin)


class MarkdownParser:
    """Parse and split markdown files intelligently"""

    @staticmethod
    def split_by_headings(content: str, max_chars: int) -> List[Tuple[str, str]]:
        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_heading = "Introduction"
        current_size = 0

        for line in lines:
            if line.startswith("#"):
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    chunks.append((current_heading, chunk_text))
                    current_chunk = []
                    current_size = 0
                current_heading = line.lstrip("#").strip()

            current_chunk.append(line)
            current_size += len(line) + 1

            if current_size > max_chars and len(current_chunk) > 10:
                chunk_text = "\n".join(current_chunk)
                chunks.append((current_heading, chunk_text))
                current_chunk = []
                current_size = 0

        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append((current_heading, chunk_text))

        return chunks

    @staticmethod
    def split_by_size(content: str, max_chars: int) -> List[Tuple[str, str]]:
        chunks = []
        start = 0
        chunk_num = 0

        while start < len(content):
            end = min(start + max_chars, len(content))
            if end < len(content):
                last_space = content.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append((f"Section {chunk_num + 1}", chunk_text))
                chunk_num += 1
            start = end + 1

        return chunks

    @classmethod
    def smart_split(cls, content: str, max_tokens: int) -> List[Tuple[str, str]]:
        max_chars = TokenEstimator.safe_chunk_size(max_tokens)
        heading_chunks = cls.split_by_headings(content, max_chars)

        final_chunks = []
        for heading, chunk_text in heading_chunks:
            if TokenEstimator.estimate(chunk_text) > max_tokens:
                sub_chunks = cls.split_by_size(chunk_text, max_chars)
                for i, (_, sub_text) in enumerate(sub_chunks):
                    final_chunks.append((f"{heading} (part {i + 1})", sub_text))
            else:
                final_chunks.append((heading, chunk_text))

        return final_chunks


class LLamaClient:
    """Client for llama.cpp server"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.available = False

    def check_availability(self) -> bool:
        try:
            import requests

            response = requests.get(f"{self.base_url}/health", timeout=5)
            self.available = response.status_code == 200
            return self.available
        except:
            print("⚠️  llama.cpp server not available - using mock mode")
            self.available = False
            return False

    def generate(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.3
    ) -> str:
        if not self.available:
            return self._mock_generate(prompt, max_tokens)

        try:
            import requests

            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": ["</s>", "User:", "Assistant:"],
            }

            response = requests.post(
                f"{self.base_url}/completion", json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("content", "")
            else:
                return self._mock_generate(prompt, max_tokens)
        except Exception as e:
            print(f"Error calling llama.cpp: {e}")
            return self._mock_generate(prompt, max_tokens)

    def _mock_generate(self, prompt: str, max_tokens: int) -> str:
        if "summarize" in prompt.lower():
            lines = prompt.split("\n")
            section_name = "the content"
            for line in lines:
                if "section:" in line.lower():
                    section_name = line.split(":")[1].strip()
                    break
            return f"Summary of {section_name}: Key points include technical details, practical applications, and best practices outlined in this section."
        elif "combine" in prompt.lower():
            return "Combined Summary: The document covers core concepts, practical applications, technical details, and recommendations."
        elif "validate" in prompt.lower():
            return "Validation Results: Coverage Assessment: ✓ PASS (85% coverage)"
        return "Generated summary content."


class PlannerAgent:
    def __init__(self, llm_client: LLamaClient):
        self.llm = llm_client

    def analyze_structure(self, file_path: str, content: str) -> Dict:
        lines = content.split("\n")
        heading_counts = {}
        for line in lines:
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                heading_counts[level] = heading_counts.get(level, 0) + 1

        word_count = len(content.split())
        return {
            "file_path": file_path,
            "word_count": word_count,
            "heading_levels": heading_counts,
            "has_code_blocks": "```" in content,
            "estimated_tokens": TokenEstimator.estimate(content),
            "complexity": "high"
            if word_count > 2000
            else "medium"
            if word_count > 500
            else "low",
        }

    def create_plan(self, metadata: Dict) -> Dict:
        estimated_tokens = metadata["estimated_tokens"]
        if estimated_tokens <= 8000:
            strategy = "direct"
            num_chunks = 1
        else:
            strategy = "chunked"
            num_chunks = math.ceil(estimated_tokens / 7000)

        return {
            "strategy": strategy,
            "num_chunks": num_chunks,
            "requires_validation": estimated_tokens > 5000,
        }


class ExecutorAgent:
    def __init__(self, llm_client: LLamaClient):
        self.llm = llm_client
        self.parser = MarkdownParser()

    def process_file(self, file_path: str, config: AgentConfig) -> FileSummary:
        print(f"\n📄 Processing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_summary = FileSummary(
            file_path=file_path,
            metadata={
                "size_bytes": os.path.getsize(file_path),
                "total_tokens": TokenEstimator.estimate(content),
            },
        )

        chunks = self.parser.smart_split(content, config.max_tokens_per_chunk)
        print(f"   Split into {len(chunks)} chunk(s)")

        for i, (heading, chunk_text) in enumerate(chunks):
            print(f"   → Summarizing chunk {i + 1}/{len(chunks)}: {heading[:50]}...")
            summary = self._summarize_chunk(heading, chunk_text, i, file_path, config)
            file_summary.summaries.append(summary)

        if len(file_summary.summaries) > 1:
            print(f"   → Combining {len(file_summary.summaries)} chunk summaries...")
            file_summary.combined_summary = self._combine_summaries(
                file_summary.summaries, config
            )
        elif len(file_summary.summaries) == 1:
            file_summary.combined_summary = file_summary.summaries[0].content_summary

        print(
            f"   ✓ File summary complete ({len(file_summary.combined_summary)} chars)"
        )
        return file_summary

    def _summarize_chunk(
        self,
        heading: str,
        content: str,
        chunk_idx: int,
        file_path: str,
        config: AgentConfig,
    ) -> ChunkSummary:
        prompt = f"""You are an expert technical writer. Summarize the following markdown section concisely.

SECTION: {heading}
FILE: {os.path.basename(file_path)}

CONTENT:
{content[: config.max_tokens_per_chunk]}

INSTRUCTIONS:
- Focus on key points and main ideas
- Preserve technical accuracy
- Use bullet points for clarity
- Keep summary under 500 words

SUMMARY:"""

        summary_text = self.llm.generate(
            prompt, max_tokens=config.max_output_tokens, temperature=config.temperature
        )

        return ChunkSummary(
            chunk_id=f"{hashlib.md5(content.encode()).hexdigest()[:8]}",
            content_summary=summary_text.strip(),
            source_file=file_path,
            chunk_index=chunk_idx,
            token_estimate=TokenEstimator.estimate(summary_text),
        )

    def _combine_summaries(
        self, summaries: List[ChunkSummary], config: AgentConfig
    ) -> str:
        all_summaries = "\n\n".join(
            [
                f"=== Chunk {i + 1} ===\n{s.content_summary}"
                for i, s in enumerate(summaries)
            ]
        )

        prompt = f"""You are an expert editor. Combine the following chunk summaries into a cohesive document summary.

CHUNK SUMMARIES:
{all_summaries[:7000]}

INSTRUCTIONS:
- Synthesize information across all chunks
- Eliminate redundancy
- Organize by importance
- Keep under 800 words

COMBINED SUMMARY:"""

        return self.llm.generate(
            prompt, max_tokens=config.max_output_tokens, temperature=config.temperature
        ).strip()


class ValidatorAgent:
    def __init__(self, llm_client: LLamaClient):
        self.llm = llm_client

    def validate_summary(
        self, original_content: str, summary: str, config: AgentConfig
    ) -> Tuple[bool, str]:
        prompt = f"""Evaluate this summary against the original content.

ORIGINAL CONTENT (first 3000 chars):
{original_content[:3000]}

SUMMARY TO VALIDATE:
{summary}

Provide coverage percentage and Pass/Fail verdict.

VALIDATION RESULT:"""

        validation_result = self.llm.generate(prompt, max_tokens=500, temperature=0.1)
        is_valid = "pass" in validation_result.lower() or "80%" in validation_result

        return is_valid, validation_result


class Orchestrator:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm_client = LLamaClient(config.llama_server_url)
        self.planner = PlannerAgent(self.llm_client)
        self.executor = ExecutorAgent(self.llm_client)
        self.validator = ValidatorAgent(self.llm_client)
        self.llm_client.check_availability()

    def summarize_directory(self, target_dir: str) -> DirectorySummary:
        print(f"\n{'=' * 70}")
        print(f"🚀 Starting Multi-Agent Markdown Summarization")
        print(f"{'=' * 70}")
        print(f"Target Directory: {target_dir}")
        print(f"Mode: {'LIVE' if self.llm_client.available else 'MOCK'}")
        print(f"{'=' * 70}\n")

        md_files = list(Path(target_dir).rglob("*.md"))
        print(f"📁 Found {len(md_files)} markdown file(s)\n")

        if not md_files:
            return DirectorySummary(dir_path=target_dir)

        dir_summary = DirectorySummary(dir_path=target_dir)
        start_time = time.time()

        for i, md_file in enumerate(md_files, 1):
            print(f"\n[{i}/{len(md_files)}] ", end="")
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                metadata = self.planner.analyze_structure(str(md_file), content)
                plan = self.planner.create_plan(metadata)
                print(
                    f"Plan: {plan['strategy']} strategy, {plan['num_chunks']} chunk(s)"
                )

                file_summary = self.executor.process_file(str(md_file), self.config)

                if plan.get("requires_validation", False):
                    print(f"   → Validating summary quality...")
                    is_valid, _ = self.validator.validate_summary(
                        content, file_summary.combined_summary, self.config
                    )
                    print(f"   ✓ Validation {'passed' if is_valid else 'failed'}")

                dir_summary.file_summaries.append(file_summary)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"📊 Generating Final Combined Summary...")
        print(f"{'=' * 70}\n")

        dir_summary.combined_summary = self._generate_final_summary(dir_summary)
        print(
            f"\n✅ Complete! Processed {len(dir_summary.file_summaries)} files in {elapsed:.1f}s"
        )

        return dir_summary

    def _generate_final_summary(self, dir_summary: DirectorySummary) -> str:
        all_summaries = []
        for fs in dir_summary.file_summaries:
            filename = os.path.basename(fs.file_path)
            all_summaries.append(
                f"FILE: {filename}\nSUMMARY:\n{fs.combined_summary[:1000]}\n"
            )

        combined_text = "\n".join(all_summaries)[:8000]

        prompt = f"""Create a comprehensive executive summary of all provided markdown files.

FILE SUMMARIES:
{combined_text}

INSTRUCTIONS:
- Synthesize across all files
- Highlight most important concepts
- Organize by topic area
- Keep under 1500 words

EXECUTIVE SUMMARY:"""

        return self.llm_client.generate(
            prompt,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
        ).strip()

    def save_results(
        self, dir_summary: DirectorySummary, output_dir: str = "./summaries"
    ):
        os.makedirs(output_dir, exist_ok=True)

        for fs in dir_summary.file_summaries:
            filename = os.path.basename(fs.file_path).replace(".md", "_summary.md")
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Summary: {os.path.basename(fs.file_path)}\n\n")
                f.write(f"**Source:** {fs.file_path}\n\n---\n\n{fs.combined_summary}")
            print(f"💾 Saved: {output_path}")

        combined_path = os.path.join(output_dir, "COMBINED_SUMMARY.md")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(
                f"# Combined Summary\n\n**Directory:** {dir_summary.dir_path}\n**Files:** {len(dir_summary.file_summaries)}\n\n---\n\n{dir_summary.combined_summary}"
            )
        print(f"💾 Saved combined summary: {combined_path}")


def create_demo_files(demo_dir: str = "./demo_docs"):
    os.makedirs(demo_dir, exist_ok=True)

    files = {
        "01_getting_started.md": """# Getting Started with Python

## Introduction
Python is a versatile programming language known for simplicity and readability.

## Installation
Visit python.org to download and install the latest version.

## Basic Syntax
Python uses dynamic typing and indentation for code blocks.

```python
name = "Alice"
age = 30
```

## Best Practices
1. Follow PEP 8 guidelines
2. Write meaningful docstrings
3. Use virtual environments
""",
        "02_advanced_topics.md": """# Advanced Python Concepts

## Decorators
Decorators modify function behavior without changing code.

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

## Generators
Generators provide memory-efficient iteration.

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
```

## Performance Optimization
Use profiling tools like cProfile and optimize data structures.
""",
        "03_project_structure.md": """# Python Project Structure

## Recommended Layout
Organize projects with src/, tests/, docs/ directories.

## Configuration Files
Use requirements.txt or pyproject.toml for dependencies.

## Testing Strategy
Write unit tests with pytest and follow AAA pattern.

## Documentation
Use Google or NumPy style docstrings for clarity.
""",
    }

    for filename, content in files.items():
        filepath = os.path.join(demo_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Created: {filepath}")

    print(f"\n✅ Created {len(files)} demo files in {demo_dir}/")
    return demo_dir


def main():
    print("Creating demo markdown files...\n")
    demo_dir = create_demo_files()

    config = AgentConfig(
        max_tokens_per_chunk=8000,
        max_output_tokens=1200,
        llama_server_url="http://localhost:8080",
        temperature=0.3,
    )

    orchestrator = Orchestrator(config)
    dir_summary = orchestrator.summarize_directory(demo_dir)

    print("\n" + "=" * 70)
    print("💾 Saving Results...")
    print("=" * 70 + "\n")
    orchestrator.save_results(dir_summary, output_dir="./demo_summaries")

    print("\n" + "=" * 70)
    print("📋 FINAL COMBINED SUMMARY")
    print("=" * 70 + "\n")
    print(dir_summary.combined_summary)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
