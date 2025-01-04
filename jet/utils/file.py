import os


def search_files(
    base_dir: str | list[str],
    extensions: list[str],
    include_files: list[str] = [],
    exclude_files: list[str] = [],
) -> list[str]:
    """
    Scrape directories for files matching specific extensions and inclusion/exclusion criteria.

    Args:
        base_dir (str | list[str]): Base directory or list of directories to search.
        extensions (list[str]): File extensions to include.
        include_files (list[str], optional): Files to include. Defaults to [].
        exclude_files (list[str], optional): Files to exclude. Defaults to [].

    Returns:
        list[str]: List of filtered file paths.
    """
    if isinstance(base_dir, str):
        base_dirs = [base_dir]
    else:
        base_dirs = base_dir

    files = []
    for dir_path in base_dirs:
        files.extend(
            os.path.join(root, f)
            for root, _, filenames in os.walk(dir_path)
            for f in filenames
            if any(f.endswith(ext) for ext in extensions)
        )

    if include_files:
        files = [file for file in files if any(
            inc in file for inc in include_files)]
    if exclude_files:
        files = [file for file in files if not any(
            exc in file for exc in exclude_files)]

    return files
