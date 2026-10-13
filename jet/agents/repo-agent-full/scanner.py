"""
Repository scanner that identifies supported files and reads content.
"""

from pathlib import Path
from typing import List, Tuple

from config import chunking
from utils import logger


class RepositoryScanner:
    def __init__(self):
        self.supported_exts = chunking.SUPPORTED_EXTS

    def scan(self, repo_path: str) -> List[Tuple[str, str, str]]:
        """
        Scans repository and returns list of (relative_path, language, content).
        """
        logger.info(f"Scanning repository: {repo_path}")
        files = []
        repo_dir = Path(repo_path)

        if not repo_dir.exists():
            raise FileNotFoundError(f"Path not found: {repo_path}")

        for file_path in repo_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_exts:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        rel_path = str(file_path.relative_to(repo_dir))
                        lang = self._get_lang(file_path.suffix)
                        files.append((rel_path, lang, content))
                except Exception as e:
                    logger.warning(f"Skipped {file_path}: {e}")

        logger.info(f"Found {len(files)} supported files.")
        return files

    def _get_lang(self, ext: str) -> str:
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".java": "java",
            ".go": "go",
        }
        return mapping.get(ext.lower(), "text")
