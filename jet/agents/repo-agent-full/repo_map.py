"""
Repository Map generator for structural awareness.
Supports Python (AST), Markdown (Headers), and generic file listing.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List

from config import chunking
from utils import logger


class RepoMap:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.map_data: Dict = {
            "root": self.repo_path.name,
            "structure": {},
            "symbols": {},
        }
        # Use the same extensions defined in the scanner/chunker config
        self.supported_exts = [ext.lower() for ext in chunking.SUPPORTED_EXTS]

    def generate(self) -> Dict:
        """
        Scan the repository and build a structural map.
        """
        logger.info("Generating repository map...")

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                suffix = file_path.suffix.lower()
                if suffix in self.supported_exts:
                    rel_path = str(file_path.relative_to(self.repo_path))
                    self._process_file(rel_path, file_path, suffix)

        logger.info(
            f"Repository map generated with {len(self.map_data['symbols'])} indexed files."
        )
        return self.map_data

    def _process_file(self, rel_path: str, full_path: Path, suffix: str):
        """Extract top-level symbols or structure based on file type."""
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                source = f.read()

            symbols = []

            if suffix == ".py":
                symbols = self._extract_python_symbols(source)
            elif suffix in [".md", ".mdx"]:
                symbols = self._extract_markdown_headers(source)
            else:
                # For other code/text files, just note their existence
                # or extract simple patterns if needed
                symbols = [f"File: {full_path.name}"]

            if symbols:
                self.map_data["symbols"][rel_path] = symbols

            # Update directory structure
            parts = rel_path.split("/")
            current = self.map_data["structure"]
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = None

        except Exception as e:
            logger.debug(f"Could not parse {rel_path} for map: {e}")

    def _extract_python_symbols(self, source: str) -> List[str]:
        """Extract top-level classes and functions from Python code."""
        symbols = []
        try:
            tree = ast.parse(source)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(f"def {node.name}")
                elif isinstance(node, ast.ClassDef):
                    symbols.append(f"class {node.name}")
        except SyntaxError:
            pass
        return symbols

    def _extract_markdown_headers(self, source: str) -> List[str]:
        """Extract top-level headers from Markdown files."""
        symbols = []
        for line in source.splitlines():
            if line.startswith("#"):
                # Limit to H1 and H2 to keep map concise
                if line.startswith("## ") or line.startswith("# "):
                    symbols.append(line.strip())
        return symbols

    def get_summary_for_query(self, query: str) -> str:
        """
        Return a string representation of the map relevant to a query.
        """
        if not self.map_data or not self.map_data["symbols"]:
            return "No repository map available."

        summary_parts = []
        summary_parts.append("## Project Structure")
        # Limit structure depth if it's too large, but for now dump it
        summary_parts.append(json.dumps(self.map_data["structure"], indent=2))

        if self.map_data["symbols"]:
            summary_parts.append("\n## Key Content by File")
            # Show first 20 files to avoid overwhelming context
            for file, syms in list(self.map_data["symbols"].items())[:20]:
                sym_list = ", ".join(syms[:5])  # Limit symbols per file
                if len(syms) > 5:
                    sym_list += "..."
                summary_parts.append(f"- `{file}`: {sym_list}")

        return "\n".join(summary_parts)
