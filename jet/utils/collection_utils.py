from collections import defaultdict
from typing import List, Dict, Any, Union


def group_by(data: List[Union[Dict[str, Any], object]], key: str) -> Dict[Any, List[Any]]:
    grouped = defaultdict(list)
    for item in data:
        if isinstance(item, dict):
            value = item.get(key)
        else:
            value = getattr(item, key, None)
        grouped[value].append(item)
    return dict(grouped)
