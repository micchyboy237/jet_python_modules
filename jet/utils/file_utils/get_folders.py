from pathlib import Path
from typing import List


def get_folder_absolute_paths(base_dir: str, depth: int = 1) -> List[str]:
    """
    Retrieves absolute paths of all directories at the specified depth under the base directory.
    Only directories at the exact depth are returned, excluding lower-depth directories.
    Depth 1 means immediate subdirectories, depth 2 means subdirectories of those, and so on.

    Args:
        base_dir (str): The path to the base directory.
        depth (int): The depth level to retrieve directories from (default is 1).

    Returns:
        List[str]: A list of absolute paths to directories at the specified depth.

    Raises:
        FileNotFoundError: If the base directory does not exist.
        NotADirectoryError: If the base path is not a directory.
        ValueError: If depth is less than 1.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    base_path = Path(base_dir).resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")
    if not base_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_dir}")

    result: List[str] = []

    # Start with immediate subdirectories
    current_paths = [base_path]

    # Iterate to the specified depth
    for current_depth in range(1, depth + 1):
        next_paths = []
        for path in current_paths:
            # Collect directories at the current level
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            if current_depth == depth:
                # At the target depth, add absolute paths to result
                result.extend(str(p.resolve()) for p in subdirs)
            else:
                # Not at target depth, collect subdirectories for next iteration
                next_paths.extend(subdirs)
        current_paths = next_paths

    return result


def main():
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python get_folders.py <directory_path> [depth]")
        sys.exit(1)
    try:
        depth = int(sys.argv[2]) if len(sys.argv) == 3 else 1
        folder_paths = get_folder_absolute_paths(sys.argv[1], depth)
        for path in folder_paths:
            print(path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
