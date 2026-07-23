import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from jet.adapters.llama_cpp.llm_utils import chat

# =====================================================================
# Logging Configuration
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DocSummarizer")

# =====================================================================
# Configuration & Context Window Budgeting (10k Tokens Total)
# =====================================================================
MAX_CONTEXT = 10240
SYSTEM_PROMPT_BUDGET = 800
MAX_OUTPUT_TOKENS = 1200
# Safe input payload limit for any single LLM request:
MAX_INPUT_TOKENS = MAX_CONTEXT - SYSTEM_PROMPT_BUDGET - MAX_OUTPUT_TOKENS  # 8240 tokens


# Rough character-to-token ratio estimation (4 characters = 1 token)
def estimate_tokens(text: str) -> int:
    return len(text) // 4


def extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """Cleans markdown formatting and parses JSON safely."""
    clean_text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    clean_text = re.sub(r"\s*```$", "", clean_text.strip(), flags=re.MULTILINE)

    # Extract JSON string if wrapped in extra prose
    json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
    if json_match:
        clean_text = json_match.group(0)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}\nRaw Content:\n{raw_text}")
        return {
            "error": "Failed to parse JSON output",
            "raw_text": raw_text,
        }


# =====================================================================
# Agent Prompt Definitions
# =====================================================================
SYSTEM_PROMPT = """You are a technical documentation summarization agent.
Your task is to analyze raw markdown text and produce concise, structured json outputs.
Return ONLY a valid JSON object matching the requested schema. Do NOT include markdown fences around the JSON."""

LEAF_USER_PROMPT = """Analyze this markdown document (Path: {file_path}).
Extract the core technical purpose, key components, and vital configurations.

```markdown
{content}

```

Return JSON with this schema:
{{
"title": "Document Title or Primary Subject",
"summary": "2-3 sentence overview",
"key_topics": ["topic 1", "topic 2"],
"code_highlights": ["important CLI command, config key, or exported API"]
}}"""

DIRECTORY_USER_PROMPT = """Analyze the following JSON summaries of files located inside the folder '{dir_path}'.
Synthesize them into a single coherent folder-level summary.

Files summarized:
{summaries_json}

Return JSON with this schema:
{{
"directory": "{dir_path}",
"folder_purpose": "Primary responsibility of this directory",
"highlights": ["key feature or document 1", "key feature 2"]
}}"""

# =====================================================================

# LLM Execution Interface using chat(..., stream=True)

# =====================================================================


async def call_llm_chat(user_prompt: str, context_label: str = "") -> Dict[str, Any]:
    """Wrapper calling jet.llm.llm_utils.chat with stream=True and logging."""
    logger.info(f"[{context_label}] Initiating LLM chat request (stream=True)...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Call chat with streaming enabled
        response_stream = chat(messages, stream=True)
        accumulated_text = ""

        if hasattr(response_stream, "__iter__") or hasattr(
            response_stream, "__aiter__"
        ):
            # Check if it's async iterable
            if asyncio.iscoroutinefunction(
                getattr(response_stream, "__anext__", None)
            ) or hasattr(response_stream, "__aiter__"):
                async for chunk in response_stream:
                    delta = getattr(chunk, "content", str(chunk))
                    accumulated_text += delta
            else:
                for chunk in response_stream:
                    delta = getattr(chunk, "content", str(chunk))
                    accumulated_text += delta
        else:
            accumulated_text = str(response_stream)

        logger.info(
            f"[{context_label}] Stream complete. Received {len(accumulated_text)} characters (~{estimate_tokens(accumulated_text)} tokens)."
        )
        logger.debug(f"[{context_label}] Raw output:\n{accumulated_text}")

        parsed_json = extract_json_payload(accumulated_text)
        return parsed_json

    except Exception as e:
        logger.exception(
            f"[{context_label}] Exception occurred during chat completion: {e}"
        )
        return {"error": str(e)}


# =====================================================================

# File Processing & Splitter Utilities

# =====================================================================


def read_and_chunk_markdown(file_path: Path) -> List[str]:
    """Reads a markdown file and chunks it by top-level headers (##) if it exceeds token limit."""
    logger.info(f"Reading file: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    tokens = estimate_tokens(content)

    if tokens <= MAX_INPUT_TOKENS:
        logger.info(
            f"File {file_path.name} fits within limit ({tokens} tokens). No chunking required."
        )
        return [content]

    logger.warning(
        f"File {file_path.name} exceeds max input tokens ({tokens} > {MAX_INPUT_TOKENS}). Chunking..."
    )

    sections = re.split(r"\n(?=#{1,3}\s)", content)
    chunks = []
    current_chunk = ""

    for section in sections:
        if estimate_tokens(current_chunk + section) > MAX_INPUT_TOKENS:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section
        else:
            current_chunk += "\n" + section

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(
        f"File {file_path.name} successfully split into {len(chunks)} chunk(s)."
    )
    return chunks


# =====================================================================

# Recursive Directory Agent Manager

# =====================================================================


class RecursiveDocSummarizer:
    def __init__(self, concurrency_limit: int = 4):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        logger.info(
            f"Initialized RecursiveDocSummarizer with concurrency limit = {concurrency_limit}"
        )

    async def _summarize_leaf_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file (handling multi-chunk files if necessary)."""
        async with self.semaphore:
            logger.info(f"Starting leaf processing for: {file_path}")
            chunks = read_and_chunk_markdown(file_path)
            chunk_summaries = []

            for idx, chunk in enumerate(chunks):
                label = f"Leaf:{file_path.name}-Chunk-{idx + 1}/{len(chunks)}"
                prompt = LEAF_USER_PROMPT.format(
                    file_path=str(file_path), content=chunk
                )
                res = await call_llm_chat(prompt, context_label=label)
                chunk_summaries.append(res)

            # Consolidate chunks if file was split
            if len(chunk_summaries) == 1:
                final_res = chunk_summaries[0]
            else:
                logger.info(
                    f"Consolidating {len(chunk_summaries)} chunk summaries for: {file_path.name}"
                )
                final_res = {
                    "title": chunk_summaries[0].get("title", file_path.name),
                    "summary": " ".join(
                        [
                            c.get("summary", "")
                            for c in chunk_summaries
                            if isinstance(c, dict)
                        ]
                    ),
                    "key_topics": list(
                        set(
                            t
                            for c in chunk_summaries
                            if isinstance(c, dict)
                            for t in c.get("key_topics", [])
                        )
                    ),
                    "code_highlights": [
                        ch
                        for c in chunk_summaries
                        if isinstance(c, dict)
                        for ch in c.get("code_highlights", [])
                    ],
                }

            final_res["file_path"] = str(file_path)
            logger.info(f"Completed leaf summary for: {file_path}")
            return final_res

    async def process_directory(self, target_dir: Path) -> Dict[str, Any]:
        """Recursively walks directory tree bottom-up, aggregating file summaries into directory summaries."""
        logger.info(f"Scanning target directory: {target_dir.resolve()}")
        dir_summary_map: Dict[str, Any] = {}

        # 1. Gather all file tasks grouped by subdirectories
        dir_to_files: Dict[Path, List[Path]] = {}
        for root, _, files in os.walk(target_dir):
            md_files = [
                Path(root) / f for f in files if f.endswith((".md", ".markdown"))
            ]
            if md_files:
                dir_to_files[Path(root)] = md_files

        if not dir_to_files:
            logger.warning("No markdown files found in the specified path.")
            return {"error": "No markdown files found."}

        logger.info(
            f"Discovered {len(dir_to_files)} directories containing markdown files."
        )

        # 2. Process all Leaf Files concurrently across the entire workspace
        all_leaf_tasks = []
        file_to_task_idx = {}
        idx = 0

        for folder, files in dir_to_files.items():
            for f in files:
                all_leaf_tasks.append(self._summarize_leaf_file(f))
                file_to_task_idx[f] = idx
                idx += 1

        logger.info(
            f"Triggering concurrent processing for {len(all_leaf_tasks)} leaf markdown document(s)..."
        )
        leaf_results = await asyncio.gather(*all_leaf_tasks)
        logger.info("All leaf markdown documents successfully summarized.")

        # 3. Bottom-Up Folder Synthesis (Sort by depth descending)
        sorted_dirs = sorted(
            dir_to_files.keys(), key=lambda p: len(p.parts), reverse=True
        )

        logger.info("Starting Bottom-Up folder synthesis...")
        for folder in sorted_dirs:
            folder_str = str(folder)
            logger.info(f"Synthesizing folder level summary for: {folder_str}")

            folder_file_summaries = [
                leaf_results[file_to_task_idx[f]] for f in dir_to_files[folder]
            ]

            summaries_prompt_payload = json.dumps(folder_file_summaries, indent=2)

            # Safeguard context size if folder has dozens of files
            if estimate_tokens(summaries_prompt_payload) > MAX_INPUT_TOKENS:
                logger.warning(
                    f"Folder summary payload for {folder_str} exceeds token budget. Truncating."
                )
                summaries_prompt_payload = summaries_prompt_payload[
                    : MAX_INPUT_TOKENS * 4
                ]

            prompt = DIRECTORY_USER_PROMPT.format(
                dir_path=folder_str, summaries_json=summaries_prompt_payload
            )

            async with self.semaphore:
                dir_summary = await call_llm_chat(
                    prompt, context_label=f"DirSummary:{folder.name}"
                )
                dir_summary["file_count"] = len(folder_file_summaries)
                dir_summary_map[folder_str] = dir_summary

        # 4. Final Root Overview Synthesis
        logger.info("Synthesizing final executive root overview...")
        root_summary_payload = json.dumps(list(dir_summary_map.values()), indent=2)
        final_prompt = f"Synthesize these folder summaries into an executive root README structure:\n{root_summary_payload}"

        async with self.semaphore:
            final_root_summary = await call_llm_chat(
                final_prompt, context_label="RootSynthesis"
            )

        logger.info("Hierarchical directory summarization completed successfully.")
        return {
            "root_overview": final_root_summary,
            "directory_tree": dir_summary_map,
            "leaf_documents": leaf_results,
        }


# =====================================================================

# Demo Environment Setup Helper

# =====================================================================


def setup_mock_directory_tree(base_path: Path):
    """Generates a temporary nested documentation folder with dummy markdown files."""
    logger.info(f"Creating mock documentation tree at: {base_path.resolve()}")
    base_path.mkdir(parents=True, exist_ok=True)

    docs = {
        "index.md": "# System Architecture\nThis repo handles microservices deployment.",
        "api/endpoints.md": "## API Endpoints\n### POST /v1/chat\nSends completion prompts to the model.",
        "api/auth.md": "## Authentication\nSet `BEARER_TOKEN` environment variable to secure local routes.",
        "deploy/docker.md": "## Docker Engine\nUse `docker-compose up --build` to spin up local LLM instances.",
    }

    for rel_path, content in docs.items():
        file_path = base_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        logger.info(f"Created mock file: {file_path}")


# =====================================================================

# Main Execution CLI Entry Point

# =====================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Markdown Tree Summarizer using jet.llm chat"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="./demo_docs",
        help="Target directory containing markdown files",
    )
    parser.add_argument(
        "--parallel", type=int, default=4, help="Number of concurrent agent slots"
    )
    args = parser.parse_args()

    target_path = Path(args.dir)

    # If demo directory doesn't exist, create it automatically
    if not target_path.exists() and args.dir == "./demo_docs":
        setup_mock_directory_tree(target_path)

    summarizer = RecursiveDocSummarizer(concurrency_limit=args.parallel)

    logger.info("==========================================================")
    logger.info("       STARTING HIERARCHICAL RECURSIVE SUMMARIZER         ")
    logger.info(f" Context Window Ceiling : {MAX_CONTEXT} tokens")
    logger.info(f" Concurrency Limit      : {args.parallel} slots")
    logger.info("==========================================================")

    results = await summarizer.process_directory(target_path)

    print("\n================ FINAL SYSTEM SUMMARY ================")
    print(json.dumps(results.get("root_overview", {}), indent=2))

    print("\n================ FOLDER BREAKDOWN ====================")
    for folder, data in results.get("directory_tree", {}).items():
        print(f"\n📂 [{folder}] ({data.get('file_count', 0)} files)")
        print(f"   Purpose   : {data.get('folder_purpose', 'N/A')}")
        print(f"   Highlights: {', '.join(data.get('highlights', []))}")


if __name__ == "__main__":
    asyncio.run(main())
