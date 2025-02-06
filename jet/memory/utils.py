from collections import defaultdict
from typing import List, Dict, Any, Tuple


def combine_paths(sample_arr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for entry in sample_arr:
        if "id" in entry:
            results.append({
                "_id": entry["id"],
                "_labels": entry["labels"],
                "_label": entry["labels"][-1],
                **entry["properties"],
            })
        else:
            source = entry.get("source", {})
            action = entry.get("action", "")
            targets = entry.get("targets", [])

            # Check if the path is valid and has at least 3 elements
            results.append({
                "source": {
                    "_id": source["id"],
                    "_labels": source["labels"],
                    "_label": source["labels"][-1],
                    **source["properties"],
                },
                "action": action,
                "targets": [{
                    "_id": target["id"],
                    "_labels": target["labels"],
                    "_label": target["labels"][-1],
                    **target["properties"],
                } for target in targets]
            })

    return results
