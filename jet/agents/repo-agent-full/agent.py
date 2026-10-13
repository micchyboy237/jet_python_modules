"""
Agentic workflow implementation using jet's observed LLM utilities.
Now includes Intent Routing and Repository Mapping.
"""

from typing import Dict, List, Optional

from config import models
from intent_router import IntentRouter
from jet.adapters.llama_cpp.llm_utils_observed import achat
from jet_telemetry import agent
from repo_map import RepoMap
from retriever import Retriever
from tools import RepoTools
from utils import logger


class CodeAgent:
    def __init__(self, repo_path: Optional[str] = None):
        self.retriever = Retriever()
        self.tools_instance = RepoTools(repo_path) if repo_path else None
        self.repo_path = repo_path
        self.router = IntentRouter()
        self.repo_map = RepoMap(repo_path) if repo_path else None
        self.map_content = ""

    def load_repository_map(self):
        """Generate the map after indexing is complete."""
        if self.repo_map:
            self.map_content = self.repo_map.generate()

    @agent(name="code-agent")
    async def run(self, task: str) -> Dict:
        """
        Main entry point. Routes the task based on intent.
        """
        if not self.tools_instance:
            return {"answer": "Error: No repository loaded.", "sources": []}

        # Step 1: Classify Intent
        intent = await self.router.classify(task)
        logger.info(f"Query classified as: {intent}")

        # Step 2: Prepare Context based on Intent
        extra_context = ""
        if intent in ["SUMMARY", "EXPLORATORY"]:
            extra_context = f"\n\nREPOSITORY STRUCTURE MAP:\n{self.repo_map.get_summary_for_query(task)}\n"

        # Step 3: Define Tools (Dynamic based on intent could be added here)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_context",
                    "description": "Search for relevant code snippets using semantic vector search. Best for FACTUAL and DEBUGGING intents.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the full content of a specific file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grep_search",
                    "description": "Search for exact text patterns. Best for EXPLORATORY intents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "file_ext": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

        tool_registry = {
            "retrieve_context": self._agent_retrieve,
            "read_file": self.tools_instance.read_file,
            "grep_search": self.tools_instance.grep_search,
        }

        system_prompt = (
            f"You are an expert coding assistant. The current query intent is: {intent}. "
            f"{extra_context}"
            "1. Always start by retrieving context if the question is general. "
            "2. If you need to see a whole file, use read_file. "
            "3. If you are looking for a specific symbol, use grep_search. "
            "4. Provide a final answer only when you are sure. "
            "5. Cite file paths and line numbers in your final answer."
        )

        logger.info(f"Starting agentic workflow for: '{task[:50]}...'")

        try:
            result = await achat(
                prompt_or_messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task},
                ],
                model=models.LLM_MODEL,
                tools=tools,
                tool_registry=tool_registry,
                max_tool_rounds=5,
                temperature=0.1,
                project_name="repo-agent-agentic",
            )

            sources = self._extract_sources(result.content)

            return {
                "answer": result.content,
                "sources": sources,
                "usage": result.usage,
                "intent": intent,
            }

        except Exception as e:
            logger.error(f"Agentic workflow failed: {e}")
            return {
                "answer": f"Error during agentic processing: {str(e)}",
                "sources": [],
            }

    async def _agent_retrieve(self, query: str) -> str:
        """Wrapper for retriever to fit tool interface."""
        results = await self.retriever.retrieve(query)
        if not results:
            return "No relevant context found."

        context_parts = []
        for res in results:
            part = f"File: {res['file_path']} (Lines {res['start_line']}-{res['end_line']})\n{res['content']}"
            context_parts.append(part)

        return "Retrieved Context:\n\n" + "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, content: str) -> List[Dict]:
        """Simple heuristic to extract mentioned files from the final answer."""
        sources = []
        import re

        files = re.findall(r"`?([\w\-]+\.\w+)`?", content)
        unique_files = list(set(files))
        for f in unique_files:
            if any(f.endswith(ext) for ext in [".py", ".js", ".ts", ".md"]):
                sources.append({"file": f, "lines": "N/A"})
        return sources
