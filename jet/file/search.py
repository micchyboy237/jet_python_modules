import os
import fnmatch
from typing import Generator, Optional


def traverse_directory(
    base_dir: str,
    includes: list[str],
    excludes: list[str] = [],
    limit: Optional[int] = None,
    direction: str = "forward",
    max_backward_depth: Optional[int] = None,
    max_forward_depth: Optional[int] = None  # Added max_forward_depth
) -> Generator[str, None, None]:
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
    visited_paths = set()  # Prevent circular references
    yielded_count = 0
    current_forward_depth = 0
    current_backward_depth = 0
    current_dir = os.path.abspath(base_dir)

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

    def search_dir(directory: str) -> Generator[str, None, None]:
        """Traverses a single directory and yields matching paths."""
        nonlocal yielded_count
        for root, dirs, _ in os.walk(directory, followlinks=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                real_path = os.path.realpath(folder_path)

                if real_path in visited_paths:
                    continue
                visited_paths.add(real_path)

                if match_patterns(folder_path, excludes) or any(exclude in folder_path for exclude in excludes):
                    continue
                if match_patterns(folder_path, includes) or any(include in folder_path for include in includes):
                    yield folder_path
                    yielded_count += 1
                    if limit and yielded_count >= limit:
                        return

    # Traverse forward
    if direction in {"forward", "both"}:
        while True:
            yield from search_dir(current_dir)
            if max_forward_depth is not None and current_forward_depth >= max_forward_depth:
                break
            current_forward_depth += 1
            if limit and yielded_count >= limit:
                return

    # Traverse backward
    if direction in {"backward", "both"}:
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Root directory reached
                break
            current_dir = parent_dir
            current_backward_depth += 1
            if max_backward_depth is not None and current_backward_depth > max_backward_depth:
                break
            yield from search_dir(current_dir)
            if limit and yielded_count >= limit:
                return


def main():
    base_dir = "/Users/jethroestrada/Desktop/External_Projects"  # os.getcwd()
    includes = ["<folder>/bin/activate"]
    excludes = ["<folder>/node_modules"]
    limit = 1
    direction = "forward"
    max_backward_depth = None
    max_forward_depth = 2  # Example max_forward_depth

    print("Traversing directories with the following parameters:")
    print(f"Base Directory: {base_dir}")
    print(f"Includes: {includes}")
    print(f"Excludes: {excludes}")
    print(f"Limit: {limit}")
    print(f"Direction: {direction}")
    print(f"Max Backward Depth: {max_backward_depth}")
    print(f"Max Forward Depth: {max_forward_depth}")
    print("\nMatching folders found:")

    for folder in traverse_directory(
        base_dir,
        includes,
        excludes,
        limit,
        direction,
        max_backward_depth,
        max_forward_depth
    ):
        print(folder)


if __name__ == "__main__":
    main()
