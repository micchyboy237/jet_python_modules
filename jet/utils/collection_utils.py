import re
from collections import defaultdict
from typing import Sequence, Union, Dict, Any, TypedDict, List


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
