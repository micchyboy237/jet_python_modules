"""
Repository Map generator for structural awareness.
"""

import ast
import json
from pathlib import Path
from typing import Dict

from utils import logger


class RepoMap:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.map_data: Dict = {}

    def generate(self) -> Dict:
        """
        Scan the repository and build a structural map.
        """
        logger.info("Generating repository map...")
        self.map_data = {"root": self.repo_path.name, "structure": {}, "symbols": {}}

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix == ".py":
                rel_path = str(file_path.relative_to(self.repo_path))
                self._process_python_file(rel_path, file_path)

        logger.info(
            f"Repository map generated with {len(self.map_data['symbols'])} indexed files."
        )
        return self.map_data

    def _process_python_file(self, rel_path: str, full_path: Path):
        """Extract top-level symbols from a Python file."""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)

            symbols = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class {node.name}")

            if symbols:
                self.map_data["symbols"][rel_path] = symbols

            # Add to structure
            parts = rel_path.split("/")
            current = self.map_data["structure"]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = None

        except Exception as e:
            logger.debug(f"Could not parse {rel_path} for map: {e}")

    def get_summary_for_query(self, query: str) -> str:
        """
        Return a string representation of the map relevant to a query.
        For now, returns the full structure for simplicity.
        """
        if not self.map_data:
            return "No repository map available."

        summary_parts = []
        summary_parts.append("## Project Structure")
        summary_parts.append(json.dumps(self.map_data["structure"], indent=2))

        if self.map_data["symbols"]:
            summary_parts.append("\n## Key Symbols by File")
            for file, syms in list(self.map_data["symbols"].items())[
                :10
            ]:  # Limit to top 10 for context
                summary_parts.append(f"- `{file}`: {', '.join(syms)}")

        return "\n".join(summary_parts)
