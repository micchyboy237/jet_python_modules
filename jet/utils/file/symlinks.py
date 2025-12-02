from __future__ import annotations
import shutil
from pathlib import Path
from typing import Iterable


def resolve_symlinks(base_dir: str | Path) -> None:
    """
    Resolve symlinks inside a directory OR resolve a single symlink path.

    If base_dir is:
        - a symlink: replaces it with the resolved content
        - a directory: recursively resolves all symlinks contained within
    """
    base = Path(base_dir).expanduser()

    # Allow resolving a single symlink directly
    if base.is_symlink():
        _resolve_single_symlink(base)
        return

    base = base.resolve()

    for link in _find_symlinks(base):
        _resolve_single_symlink(link)


def _find_symlinks(base: Path) -> Iterable[Path]:
    """Yield all symlinks under base directory recursively."""
    for p in base.rglob("*"):
        if p.is_symlink():
            yield p


def _resolve_single_symlink(link: Path) -> None:
    """
    Replace a single symlink with the actual file/directory
    it references, removing the original symlink.
    """
    target = link.resolve()

    if not target.exists():
        raise FileNotFoundError(f"Broken symlink: {link} -> {target}")

    link.unlink()  # remove symlink

    if target.is_file():
        shutil.copy2(target, link)
    else:
        shutil.copytree(target, link)


if __name__ == "__main__":
    resolve_symlinks("/Users/jethroestrada/.cache/huggingface/hub/models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese/snapshots/cf031e020336460d15a417eba710bbc5bb43be9a")