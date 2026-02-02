import re
import shutil
from pathlib import Path

from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.local_file_search_tool import LocalFileSearchTool
from jet.libs.smolagents.tools.local_file_write_tool import LocalFileWriteTool
from smolagents import LogLevel, ToolCallingAgent

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
WORK_DIR = OUTPUT_DIR / "code"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Local LLM configuration
# ---------------------------------------------------------------------------

model = OpenAIModel(
    model_id="qwen3-instruct-2507:4b",
    temperature=0.2,
    max_tokens=8000,
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

search_tool = LocalFileSearchTool()
write_tool = LocalFileWriteTool(work_dir=str(WORK_DIR))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = ToolCallingAgent(
    tools=[search_tool, write_tool],
    model=model,
    add_base_tools=False,
    verbosity_level=LogLevel.DEBUG,
)


# ---------------------------------------------------------------------------
# Markdown python extractor
# ---------------------------------------------------------------------------

PYTHON_BLOCK_RE = re.compile(
    r"```python\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in PYTHON_BLOCK_RE.findall(markdown)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/smolagents/jet_examples/agent_with_memory"

    # Ask agent to discover files (tool-driven)
    task = rf"""
Update files under this directory based on below documentation
Base directory: {base_dir}

Document:
### 1. `memory/long_term.py`

```python
# memory/long_term.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Optional

class LongTermMemory:
    def __init__(self, persist_dir: str = "./agent_longterm_db"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="smolagent_facts",
            metadata={{"hnsw:space": "cosine"}}
        )

    def add_fact(self, content: str, step_number: int, run_id: str = "default_run") -> str:
        if not content.strip():
            return "Empty fact ignored."
        embedding = self.embedder.encode(content).tolist()
        fact_id = f"fact-{{run_id}}-{{step_number}}-{{self.collection.count()}}"
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{{"step": step_number, "run_id": run_id, "type": "fact"}}],
            ids=[fact_id]
        )
        return f"Fact saved (id: {{fact_id}})"

    def search(self, query: str, n_results: int = 6) -> str:
        if not query.strip():
            return "No query provided."
        emb = self.embedder.encode(query).tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=n_results)
        if not res["documents"] or not res["documents"][0]:
            return "No relevant long-term facts found."
        
        lines = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            lines.append(f"• {{doc}} (step {{meta['step']}}, run {{meta.get('run_id','?')}})")
        return "\n".join(lines)


# Singleton / global instance (common pattern for agents)
long_term_memory = LongTermMemory()
```

### 2. `memory/shared_state.py`

```python
# memory/shared_state.py
import json
import os
from typing import Any, Optional

class SharedState:
    def __init__(self, file_path: str = "agent_shared_state.json"):
        self.file_path = file_path
        self.data: dict = {{}}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Warning: could not load shared state: {{e}}")

    def save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: could not save shared state: {{e}}")

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def append_to_list(self, key: str, item: Any) -> None:
        lst = self.get(key, [])
        if not isinstance(lst, list):
            lst = []
        lst.append(item)
        self.set(key, lst)

    def __repr__(self) -> str:
        return f"SharedState(keys={{list(self.data.keys())}})"


# Global instance
shared_state = SharedState()
```

### 3. `tools/memory_tools.py`

```python
# tools/memory_tools.py
from smolagents import Tool
from memory.long_term import long_term_memory
from memory.shared_state import shared_state


def save_fact(content: str) -> str:
    \"""Save important reusable fact / entity / lesson / preference to long-term memory.\"""
    if len(content) < 8:
        return "Fact too short – ignored."
    step_nr = 0
    try:
        # Best effort to get current step number
        from smolagents import CodeAgent
        agent = CodeAgent.get_current_agent()  # may not exist → fallback
        if hasattr(agent, "memory") and agent.memory.steps:
            step_nr = agent.memory.steps[-1].step_number
    except Exception:
        pass
    return long_term_memory.add_fact(content, step_nr)


def recall_facts(query: str, n_results: int = 5) -> str:
    \"""Search long-term memory for relevant past facts.\"""
    return long_term_memory.search(query, n_results)


def update_shared(key: str, value: str) -> str:
    \"""Update a value in the shared mutable state.\"""
    shared_state.set(key, value)
    return f"Shared state updated → {{key}} = {{value}}"


def read_shared(key: str) -> str:
    \"""Read current value from shared state.\"""
    v = shared_state.get(key, "<not set>")
    return f"{{key}}: {{v}}"


# Tools ready to be passed to CodeAgent
LongTermSaveTool = Tool(
    name="save_important_fact",
    description=(
        "Save concise, high-value facts, entities, goals, lessons or user preferences "
        "to long-term memory for future reuse."
    ),
    function=save_fact,
    parameters={{"content": "string"}}
)

LongTermRecallTool = Tool(
    name="recall_relevant_facts",
    description="Search long-term memory for facts relevant to the current task.",
    function=recall_facts,
    parameters={{
        "query": "string",
        "n_results": "integer (optional, default 5)"
    }}
)

SharedStateUpdateTool = Tool(
    name="update_shared_state",
    description="Update a named value in the persistent shared state.",
    function=update_shared,
    parameters={{
        "key": "string",
        "value": "string"
    }}
)

SharedStateReadTool = Tool(
    name="read_shared_state",
    description="Read a named value from the shared persistent state.",
    function=read_shared,
    parameters={{"key": "string"}}
)
```

### 4. `callbacks.py`

```python
# callbacks.py
from smolagents import ActionStep, CodeAgent
from memory.shared_state import shared_state


def auto_save_shared_state(step: ActionStep, agent: CodeAgent) -> None:
    \"""Save shared state after steps that look like final answers or big updates\"""
    if step.final_answer is not None or "saved" in str(step.observations or "").lower():
        shared_state.save()


def auto_extract_simple_facts(step: ActionStep, agent: CodeAgent) -> None:
    \"""Very naive auto-extraction – improve with LLM reflection in production\"""
    if not step.observations:
        return
    text = str(step.observations)
    if len(text) > 400:
        text = text[:380] + "..."
    keywords = ["important:", "remember:", "note that", "fact:", "key point"]
    if any(k in text.lower() for k in keywords):
        from tools.memory_tools import save_fact
        save_fact(text)
```

### 5. `agent_factory.py`

```python
# agent_factory.py
from smolagents import CodeAgent, InferenceClientModel
from tools.memory_tools import (
    LongTermSaveTool, LongTermRecallTool,
    SharedStateUpdateTool, SharedStateReadTool
)
from callbacks import auto_save_shared_state, auto_extract_simple_facts

def create_memory_enabled_agent(
    model=None,
    extra_tools=None,
    max_steps: int = 40,
    verbosity: int = 1
) -> CodeAgent:
    if model is None:
        model = InferenceClientModel()  # or your preferred model

    tools = [
        LongTermSaveTool,
        LongTermRecallTool,
        SharedStateUpdateTool,
        SharedStateReadTool,
    ]
    if extra_tools:
        tools.extend(extra_tools)

    return CodeAgent(
        tools=tools,
        model=model,
        step_callbacks=[
            auto_save_shared_state,
            auto_extract_simple_facts,
            # add more callbacks here
        ],
        max_steps=max_steps,
        verbosity_level=verbosity,
    )
```

### 6. `main.py` – usage examples

```python
# main.py
from agent_factory import create_memory_enabled_agent

if __name__ == "__main__":
    agent = create_memory_enabled_agent(verbosity=1)

    # Example 1 – basic math + manual save
    agent.run(\"""
    Compute the 15th Fibonacci number.
    Then save the result as an important fact named "fib_15_result".
    \""")

    # Example 2 – recall previous knowledge
    agent.run(\"""
    What was the 15th Fibonacci number we computed earlier?
    Use long-term memory if needed.
    \""")

    # Example 3 – shared state usage
    agent.run(\"""
    The current project name is "Aether". Save it to shared state under key "project_name".
    Then read back the project name from shared state.
    \""")

    # Example 4 – multi-turn awareness via shared state
    agent.run("What is the current project name?")
```
"""
    search_result = agent.run(task)

    print(search_result)

    # Deterministic extraction + writing
    base_path = Path(base_dir).resolve()
    output_root = Path(write_tool.work_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for md_file in base_path.rglob("*.md"):
        try:
            text = md_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        blocks = extract_python_blocks(text)
        if not blocks:
            continue

        stem = md_file.stem.replace("-", "_").replace(" ", "_")

        combined_code: list[str] = []

        for idx, code in enumerate(blocks, start=1):
            combined_code.append(
                f"# ------------------------------------------------------------------\n"
                f"# Source: {md_file.name} | Block {idx}\n"
                f"# ------------------------------------------------------------------\n"
                f"{code}\n"
            )

        # Step 1: Compute relative directory
        try:
            relative_dir = md_file.parent.relative_to(base_path)
        except ValueError:
            # md_file is not under base_path
            relative_dir = Path()  # write directly into the root

        # Step 2: Build output path using that directory
        relative_path = relative_dir / f"{stem}_example.py"

        # Step 3: Pass it to the write tool
        result = write_tool.forward(
            relative_path=str(relative_path),
            content="\n".join(combined_code),
        )

        print(result)

    print(f"\nPython files generated under: {output_root}")
