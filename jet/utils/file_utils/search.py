import os
import fnmatch
from pathlib import Path
from typing import List, Set, Optional, Iterable
from jet.logger import logger


def find_files(
    base_dir: str,
    include: List[str],
    exclude: List[str],
    include_content_patterns: List[str],
    exclude_content_patterns: List[str],
    case_sensitive: bool = False,
    extensions: List[str] = [],
    modified_after: Optional[float] = None,
) -> List[str]:
    """
    Optimized file finder with include/exclude filters, optional content matching,
    and support for double-wildcard absolute patterns.
    """

    normalized_extensions = {ext.lstrip(".").lstrip("*").lower() for ext in extensions}
    matched_files: Set[str] = set()
    base_path = Path(base_dir).resolve()

    if not base_path.exists():
        logger.warning(f"Directory does not exist: {base_dir}")
        return []

    # Normalize includes/excludes
    def normalize_patterns(patterns: List[str], is_exclude=False) -> List[str]:
        out = []
        for pat in patterns:
            if os.path.isabs(pat):
                out.append(pat)
            else:
                if pat.endswith("/") or pat.endswith("/*"):
                    pat = pat.rstrip("/") + "/**/*"
                elif not is_exclude and (pat.startswith("*/") or pat.endswith("/*/")):
                    pat = pat.strip("*/").rstrip("/") + "/**/*"
                out.append(pat)
        return out

    adjusted_include = normalize_patterns(include)
    adjusted_exclude = normalize_patterns(exclude, is_exclude=True)

    # Default: search everything
    if not adjusted_include:
        adjusted_include = ["**/*"]

    # Pre-split excludes into absolute/relative
    abs_excludes = [p for p in adjusted_exclude if os.path.isabs(p)]
    rel_excludes = [p for p in adjusted_exclude if not os.path.isabs(p)]

    def is_excluded(file_path: Path) -> bool:
        """Check if file should be excluded early (absolute + relative)."""
        f_str = str(file_path)
        # Absolute patterns
        for pat in abs_excludes:
            if "**" in pat or "*" in pat or "?" in pat:
                if fnmatch.fnmatch(f_str, pat):
                    return True
            else:
                if Path(pat) in [file_path, *file_path.parents]:
                    return True
        # Relative patterns (match from base_path)
        rel = os.path.relpath(f_str, base_path)
        for pat in rel_excludes:
            if fnmatch.fnmatch(rel, pat):
                return True
        return False

    # Collect candidates
    for pattern in adjusted_include:
        try:
            candidates: Iterable[Path]
            if os.path.isabs(pattern):
                abs_path = Path(pattern)
                if abs_path.is_file():
                    candidates = [abs_path]
                elif abs_path.is_dir():
                    candidates = abs_path.rglob("*")
                else:
                    # Handle absolute wildcard patterns
                    if any(x in pattern for x in ["*", "?", "**"]):
                        root = Path("/")
                        try:
                            candidates = root.glob(pattern.lstrip("/"))
                        except NotImplementedError:
                            # fallback: manual walk + fnmatch
                            candidates = [
                                p for p in root.rglob("*") if fnmatch.fnmatch(str(p), pattern)
                            ]
                    else:
                        continue
            else:
                candidates = base_path.rglob(pattern)

            for file_path in candidates:
                if not file_path.is_file():
                    continue

                # Exclude check (early)
                if is_excluded(file_path):
                    continue

                # Extension filter
                if normalized_extensions:
                    ext = file_path.suffix.lstrip(".").lower()
                    if ext not in normalized_extensions:
                        continue

                # Modified time filter
                if modified_after:
                    try:
                        if file_path.stat().st_mtime <= modified_after:
                            continue
                    except OSError as e:
                        logger.error(f"Failed to get modified time for {file_path}: {e}")
                        continue

                norm_path = os.path.normpath(str(file_path)).replace("/private/var", "/var")
                matched_files.add(norm_path)

        except OSError as e:
            logger.error(f"Error traversing {pattern}: {e}")

    # Final content filtering
    final_files = [
        f
        for f in matched_files
        if matches_content(f, include_content_patterns, exclude_content_patterns, case_sensitive)
    ]

    return sorted(final_files)


def matches_content(
    file_path: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    case_sensitive: bool = False,
) -> bool:
    if not include_patterns and not exclude_patterns:
        return True

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not case_sensitive:
            content = content.lower()
            include_patterns = [p.lower() for p in include_patterns]
            exclude_patterns = [p.lower() for p in exclude_patterns]

        if include_patterns and not any(
            fnmatch.fnmatch(content, p) if any(x in p for x in "*?") else p in content
            for p in include_patterns
        ):
            return False

        if exclude_patterns and any(
            fnmatch.fnmatch(content, p) if any(x in p for x in "*?") else p in content
            for p in exclude_patterns
        ):
            return False

        return True
    except (OSError, IOError) as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False
