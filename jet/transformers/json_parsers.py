import json
from typing import Any


def parse_json(obj: Any):
    if isinstance(obj, (dict, list)):
        return obj

    try:
        parsed_obj = json.loads(obj)

        return parsed_obj

    except json.JSONDecodeError:
        return obj
