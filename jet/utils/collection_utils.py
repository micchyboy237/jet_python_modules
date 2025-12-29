from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Sequence, Union, Dict, Any, TypedDict, List, Generator, TypeVar, Tuple


class GroupedResult(TypedDict):
    group: Any
    items: List[Any]


def get_nested_value(item: Union[Dict[str, Any], object], path: str) -> Any:
    path = re.sub(r"\[['\"]?([^'\"]+)['\"]?\]", r".\1", path)
    keys = path.strip(".").split(".")

    current = item
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def group_by(data: Sequence[Union[Dict[str, Any], object]], key: str) -> List[GroupedResult]:

    grouped = defaultdict(list)
    for item in data:
        value = get_nested_value(item, key)
        grouped[value].append(item)
    return [{"group": group_key, "items": items} for group_key, items in grouped.items()]


T = TypeVar("T")

def growing_windows(
    seq: Iterable[T],
    max_size: int | None = None,
    *,
    start_offset: int = 0,
    step: int = 1,
) -> Generator[Tuple[T, ...], None, None]:
    """
    Yield growing windows from a strided subsequence of `seq`.

    The subsequence starts at index `start_offset` and samples every `step` elements thereafter.
    Windows grow in size from 1 up to `max_size` (or until the subsequence is exhausted if max_size is None).

    Parameters
    ----------
    seq : Iterable[T]
        Input sequence.
    max_size : int | None, optional
        Maximum window size. If None, grows until the end of the sampled subsequence.
    start_offset : int, optional (keyword-only)
        Index of the first element to include (default 0).
    step : int, optional (keyword-only)
        Distance between consecutive sampled elements (default 1). Must be > 0.

    Yields
    ------
    Tuple[T, ...]
        Growing tuples: size 1, 2, ..., up to min(max_size, available).

    Examples
    --------
    >>> list(growing_windows([1, 2, 3, 4, 5], max_size=3, start_offset=1, step=1))
    [(2,), (2, 3), (2, 3, 4)]

    >>> list(growing_windows([1, 2, 3, 4, 5], max_size=None, start_offset=0, step=2))
    [(1,), (1, 3), (1, 3, 5)]
    """
    if step <= 0:
        return
    if max_size is not None and max_size <= 0:
        return
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")

    it = iter(seq)
    current: list[T] = []

    # Advance to the first element (start_offset)
    try:
        for _ in range(start_offset):
            next(it)
        current.append(next(it))
        yield tuple(current)
    except StopIteration:
        return

    # Continue sampling every `step` elements
    while max_size is None or len(current) < max_size:
        try:
            for _ in range(step - 1):
                next(it)
            current.append(next(it))
            yield tuple(current)
        except StopIteration:
            break