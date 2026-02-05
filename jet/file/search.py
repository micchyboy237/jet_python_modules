import os
from collections.abc import Generator

from jet.file.validation import validate_match
from jet.logger import logger

DEFAULT_EXCLUDES = [
    "**/__pycache__",
]


def traverse_directory(
    base_dir: str,
    includes: list[str],
    excludes: list[str] = [],
    limit: int | None = None,
    direction: str = "forward",
    max_backward_depth: int | None = None,
    max_forward_depth: int | None = None,
) -> Generator[tuple[str, int], None, None]:
    excludes = list(set(DEFAULT_EXCLUDES + excludes))
    visited_paths = set()
    passed_paths = set()
    yielded_count = 0
    current_dir = os.path.abspath(base_dir)
    passed_dirs_dict = {}

    if direction in {"forward", "both"} and max_forward_depth == 0:
        try:
            with os.scandir(base_dir) as it:
                for entry in it:
                    if entry.is_dir():
                        path = entry.path
                        if (
                            validate_match(path, includes, excludes)
                            and path not in visited_paths
                        ):
                            visited_paths.add(path)
                            yield path, 0
                            yielded_count += 1
                            if limit and yielded_count >= limit:
                                return
        except FileNotFoundError:
            logger.warning(f"Base directory not found: {base_dir}")
        return

    def calculate_depth(root: str) -> int:
        """Calculate the depth of a directory relative to base_dir."""
        relative_path = os.path.relpath(root, current_dir)
        if relative_path == ".":
            return 0
        return max(0, len(relative_path.split(os.sep)) - 1)

    def search_dir(
        directory: str, max_depth: int | None
    ) -> Generator[tuple[str, int], None, None]:
        nonlocal yielded_count
        for root, dirs, _ in os.walk(directory, followlinks=False):
            root_matches_patterns = validate_match(root, includes, excludes)
            if not root_matches_patterns:
                continue
            root_includes_real_path = any(
                include in root
                for include in includes
                if include.startswith("/") and os.path.exists(include)
            )
            root_excludes_real_path = any(
                exclude in root
                for exclude in excludes
                if exclude.startswith("/") and os.path.exists(exclude)
            )
            root_already_passed = any(path.startswith(root) for path in passed_paths)
            include_passed = root_matches_patterns or root_includes_real_path
            exclude_passed = root_already_passed or root_excludes_real_path
            passes = include_passed and not exclude_passed
            if passes and root != current_dir:
                current_depth = calculate_depth(root)
                if max_depth is None or current_depth <= max_depth:
                    yield root, current_depth
                    yielded_count += 1
                    passed_paths.add(root)
                    if limit and yielded_count >= limit:
                        return

    if direction in {"forward", "both"}:
        search_results_stream = search_dir(current_dir, max_forward_depth)
        depth_key = str(0)
        passed_dirs = passed_dirs_dict.get(depth_key, [])
        passed_dirs_dict[depth_key] = passed_dirs
        for folder_path, current_depth in search_results_stream:
            passed_dirs.append(folder_path)
            yield folder_path, current_depth
            if limit and yielded_count >= limit:
                return
        logger.log(
            "Depth:",
            depth_key,
            "|",
            "Dirs:",
            len(passed_dirs_dict[depth_key]),
            colors=["WHITE", "DEBUG", "GRAY", "WHITE", "SUCCESS"],
        )


def main():
    base_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts"
    includes = ["<folder>/bin/activate"]
    excludes = []
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
        max_forward_depth,
    ):
        logger.success(folder)


if __name__ == "__main__":
    main()
