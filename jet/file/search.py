import os
import fnmatch
from typing import Generator, Optional

from jet.file.validation import validate_match
from jet.logger import logger

DEFAULT_EXCLUDES = [
    "**/.git",
    "**/node_modules",
    "**/dist",
    "**/build",
    "**/public",
    "**/vector_db",
    "**/generated",
    "**/.?cache*",
    "**/__pycache__",
    "/Users/jethroestrada/Desktop/External_Projects/PortfolioBaseTemplates/firebase-resume"
]


def traverse_directory(
    base_dir: str,
    includes: list[str],
    excludes: list[str] = [],
    limit: Optional[int] = None,
    direction: str = "forward",
    max_backward_depth: Optional[int] = None,
    max_forward_depth: Optional[int] = 3
) -> Generator[tuple[str, int], None, None]:
    """
    Generator that traverses directories and yields folder paths 
    matching the include patterns but not the exclude patterns.

    :param base_dir: The directory to start traversal from.
    :param includes: Patterns to include in the search.
    :param excludes: Patterns to exclude from the search.
    :param limit: Maximum number of folder paths to yield.
    :param direction: Direction of traversal - 'forward' (default), 'backward', or 'both'.
    :param max_backward_depth: Maximum depth to traverse upwards (for 'backward' or 'both').
    :param max_forward_depth: Maximum depth to traverse downwards (for 'forward' or 'both').
    """
    excludes = list(set(DEFAULT_EXCLUDES + excludes))
    visited_paths = set()  # Prevent circular references
    passed_paths = set()  # Track passed paths
    yielded_count = 0
    base_dir = os.path.abspath(base_dir)

    def match_patterns(path: str, patterns: list[str]) -> bool:
        """Checks if a path matches any of the given patterns."""
        for pattern in patterns:
            if "<folder>" in pattern:
                folder_path = os.path.join(
                    path, pattern.replace("<folder>", "").lstrip("/"))
                if os.path.exists(folder_path):
                    return True
            elif fnmatch.fnmatch(path, f"*{os.path.normpath(pattern.lstrip('/'))}"):
                return True
        return False

    def calculate_depth(folder_path: str) -> int:
        """Calculate the depth of a folder relative to base_dir."""
        relative_path = os.path.relpath(folder_path, base_dir)
        if relative_path == ".":
            return 0
        return len([p for p in relative_path.split(os.sep) if p])

    def search_dir(directory: str) -> Generator[tuple[str, int], None, None]:
        """Traverses a single directory and yields matching paths with their depths."""
        nonlocal yielded_count
        for root, dirs, _ in os.walk(directory, followlinks=False):
            root_matches_patterns = validate_match(root, includes, excludes)

            if not root_matches_patterns:
                continue

            root_includes_real_path = any(
                include in root for include in includes
                if include.startswith("/") and os.path.exists(include)
            )
            root_excludes_real_path = any(
                exclude in root for exclude in excludes
                if exclude.startswith("/") and os.path.exists(exclude)
            )
            root_already_passed = any(
                path.startswith(root) for path in passed_paths
            )

            include_passed = root_matches_patterns or root_includes_real_path
            exclude_passed = root_already_passed or root_excludes_real_path
            passes = include_passed and not exclude_passed

            if passes:
                current_depth = calculate_depth(root)
                if max_forward_depth is None or current_depth <= max_forward_depth:
                    yield root, current_depth
                    yielded_count += 1
                    passed_paths.add(root)
                    if limit and yielded_count >= limit:
                        return

    # Traverse forward
    if direction in {"forward", "both"}:
        for folder_path, depth in search_dir(base_dir):
            yield folder_path, depth
            if limit and yielded_count >= limit:
                return

    # Traverse backward
    elif direction in {"backward", "both"}:
        current_dir = base_dir
        current_backward_depth = 0
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Root directory reached
                break
            current_dir = parent_dir
            current_backward_depth += 1
            if max_backward_depth is not None and current_backward_depth > max_backward_depth:
                break
            if match_patterns(current_dir, includes) or any(include in current_dir for include in includes):
                yield current_dir, current_backward_depth
                yielded_count += 1
            if limit and yielded_count >= limit:
                return


def main():
    # base_dir = "/Users/jethroestrada/Desktop/External_Projects"
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts"
    includes = ["<folder>/bin/activate"]
    excludes = []
    # limit = 1
    limit = None
    direction = "forward"
    max_backward_depth = None
    max_forward_depth = 3

    print("Traversing directories with the following parameters:")
    print(f"Base Directory: {base_dir}")
    print(f"Includes: {includes}")
    print(f"Excludes: {excludes}")
    print(f"Limit: {limit}")
    print(f"Direction: {direction}")
    print(f"Max Backward Depth: {max_backward_depth}")
    print(f"Max Forward Depth: {max_forward_depth}")
    logger.info("\nMatching folders found:")

    for folder in traverse_directory(
        base_dir,
        includes,
        excludes,
        limit,
        direction,
        max_backward_depth,
        max_forward_depth
    ):
        logger.success(folder)


if __name__ == "__main__":
    main()
