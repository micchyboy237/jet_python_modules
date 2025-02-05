from collections import defaultdict
from typing import List, Dict, Any, Tuple


def combine_paths(sample_arr: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str, List[Dict[str, Any]]]]:
    grouped_data = defaultdict(list)

    for entry in sample_arr:
        path = entry["path"]
        # Use frozenset to make dictionary hashable
        key = (frozenset(path[0].items()), path[1])
        grouped_data[key].append(path[2])

    result = []
    for (source_obj, relation), target_objs in grouped_data.items():
        result.append([dict(source_obj), relation, target_objs])

    return result
