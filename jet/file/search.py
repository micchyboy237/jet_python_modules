import os
import fnmatch
from typing import Generator, Optional

from jet.file.validation import validate_match
from jet.logger import logger

DEFAULT_EXCLUDES = [
    "**/__pycache__",
]


def traverse_directory(
    base_dir: str,
    includes: list[str],
    excludes: list[str] = [],
    limit: Optional[int] = None,
    direction: str = "forward",
    max_backward_depth: Optional[int] = None,
    max_forward_depth: Optional[int] = 3  # Added max_forward_depth
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
    passed_paths = set()  # Prevent circular references
    yielded_count = 0
    current_forward_depth = 0
    current_backward_depth = 0
    current_dir = os.path.abspath(base_dir)
    passed_dirs_dict = {}

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
            root_matches_patterns = validate_match(root, includes, excludes)

            if not root_matches_patterns:
                continue

            root_includes_real_path = any(include in root for include in includes
                                          if include.startswith("/") and os.path.exists(include))
            root_excludes_real_path = any(exclude in root for exclude in excludes
                                          if exclude.startswith("/") and os.path.exists(exclude))

            root_already_passed = any([path.startswith(root)
                                       for path in list(passed_paths)])

            include_passed = root_matches_patterns or root_includes_real_path
            exclude_passed = root_already_passed or root_excludes_real_path
            passes = include_passed and not exclude_passed

            if passes:
                yield root, current_forward_depth
                yielded_count += 1

                passed_paths.add(root)
                if limit and yielded_count >= limit:
                    return

            # root_passes_include = match_patterns(root, includes) or any(
            #     include in root for include in includes)

            # if root_passes_include:
            #     yield root
            #     yielded_count += 1
            #     if limit and yielded_count >= limit:
            #         return

            # for folder in dirs:
            #     folder_path = os.path.join(root, folder)
            #     real_path = os.path.realpath(folder_path)

            #     if real_path in visited_paths:
            #         continue
            #     visited_paths.add(real_path)

            #     if match_patterns(folder_path, excludes) or any(exclude in folder_path for exclude in excludes):
            #         continue
            #     if match_patterns(folder_path, includes) or any(include in folder_path for include in includes):
            #         yield folder_path
            #         yielded_count += 1
            #         if limit and yielded_count >= limit:
            #             return

    # Traverse forward
    if direction in {"forward", "both"}:
        while True:
            search_results_stream = search_dir(current_dir)
            depth_key = str(current_forward_depth)
            passed_dirs = passed_dirs_dict.get(depth_key) or []
            passed_dirs_dict[depth_key] = passed_dirs
            for folder_path, current_depth in search_results_stream:
                passed_dirs.append(folder_path)
                yield folder_path, current_depth

            if max_forward_depth is not None and current_forward_depth >= max_forward_depth:
                break

            logger.log(f"Depth:", depth_key, "|", f"Dirs:", len(
                passed_dirs_dict[depth_key]), colors=["WHITE", "DEBUG", "GRAY", "WHITE", "SUCCESS"])
            current_forward_depth += 1
            if limit and yielded_count >= limit:
                return

    # Traverse backward
    elif direction in {"backward", "both"}:
        while True:
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Root directory reached
                break
            current_dir = parent_dir
            current_backward_depth += 1
            if max_backward_depth is not None and current_backward_depth > max_backward_depth:
                break
            if match_patterns(current_dir, includes) or any(include in current_dir for include in includes):
                print("MATCHED:", current_dir)
                yield current_dir
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
