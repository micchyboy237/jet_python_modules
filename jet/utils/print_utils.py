from typing import Any, List, Optional
from collections.abc import Mapping


def get_common_dict_structure(data: List[Any]) -> dict | None:
    """
    Helper function to extract a dictionary structure with all possible keys from a list of dictionaries.
    Uses values from the first dictionary where available, otherwise from the first occurrence of the key.
    Keys are sorted alphabetically for consistent ordering.
    """
    if not data or not all(isinstance(item, Mapping) for item in data):
        return None
    all_keys = sorted(set.union(*(set(item.keys())
                      for item in data)))  # Sort keys alphabetically
    result = {}
    for key in all_keys:
        # Find the first dictionary that has this key
        for item in data:
            if key in item:
                result[key] = item[key]
                break
    return result


def print_dict_types(data: Any, prefix: str = "", indent: int = 0) -> List[str]:
    """
    Returns a list of strings describing the type of each value in a dictionary with full key paths,
    handling nested dictionaries and lists. Merges list items with identical dictionary structures
    into a single representation. Only uses numerical indices for tuples, not lists.
    Sorts dictionaries by key count in descending order.

    Args:
        data: The input data to analyze (typically a dictionary)
        prefix: The current key path prefix (used in recursion)
        indent: The current indentation level for formatting

    Returns:
        List of strings, each representing a key path and its value type
    """
    lines = []
    if isinstance(data, Mapping):
        # Sort keys by the number of keys in nested dictionaries (if applicable)
        sorted_keys = sorted(
            data.keys(),
            key=lambda k: len(data[k]) if isinstance(data[k], Mapping) else 0,
            reverse=True
        )
        for key in sorted_keys:
            value = data[key]
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, list):
                # Sort list items by key count if they are dictionaries
                sorted_list = sorted(
                    value,
                    key=lambda x: len(x) if isinstance(x, Mapping) else 0,
                    reverse=True
                ) if value else []
                common_dict = get_common_dict_structure(
                    value) if value else None
                if common_dict:
                    lines.append(f"{'  ' * indent}{new_prefix}[]: list[dict]")
                    lines.extend(print_dict_types(
                        common_dict, f"{new_prefix}[]", indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{new_prefix}: list")
                    if not value:
                        continue
                    for item in sorted_list:
                        new_item_prefix = new_prefix
                        if isinstance(item, (Mapping, tuple)):
                            lines.append(
                                f"{'  ' * (indent + 1)}{new_item_prefix}: {type(item).__name__}")
                            lines.extend(print_dict_types(
                                item, new_item_prefix, indent + 2))
                        elif isinstance(item, list):
                            lines.extend(print_dict_types(
                                item, new_item_prefix, indent + 1))
                        else:
                            lines.append(
                                f"{'  ' * (indent + 1)}{new_item_prefix}: {type(item).__name__}")
            elif isinstance(value, tuple):
                lines.append(f"{'  ' * indent}{new_prefix}: tuple")
                for index, item in enumerate(value):
                    new_tuple_prefix = f"{new_prefix}[{index}]"
                    lines.append(
                        f"{'  ' * (indent + 1)}{new_tuple_prefix}: {type(item).__name__}")
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.extend(print_dict_types(
                            item, new_tuple_prefix, indent + 2))
            elif isinstance(value, Mapping):
                lines.append(
                    f"{'  ' * indent}{new_prefix}: {type(value).__name__}")
                lines.extend(print_dict_types(value, new_prefix, indent + 1))
            else:
                lines.append(
                    f"{'  ' * indent}{new_prefix}: {type(value).__name__}")
    elif isinstance(data, list):
        # Sort list items by key count if they are dictionaries
        sorted_list = sorted(
            data,
            key=lambda x: len(x) if isinstance(x, Mapping) else 0,
            reverse=True
        ) if data else []
        common_dict = get_common_dict_structure(data) if data else None
        if common_dict:
            lines.append(f"{'  ' * indent}{prefix}[]: list[dict]")
            lines.extend(print_dict_types(
                common_dict, f"{prefix}[]", indent + 1))
        else:
            lines.append(f"{'  ' * indent}{prefix}: list")
            if not data:
                return lines
            for item in sorted_list:
                new_item_prefix = prefix
                if isinstance(item, (Mapping, tuple)):
                    lines.append(
                        f"{'  ' * (indent + 1)}{new_item_prefix}: {type(item).__name__}")
                    lines.extend(print_dict_types(
                        item, new_item_prefix, indent + 2))
                elif isinstance(item, list):
                    lines.extend(print_dict_types(
                        item, new_item_prefix, indent + 1))
                else:
                    lines.append(
                        f"{'  ' * (indent + 1)}{new_item_prefix}: {type(item).__name__}")
    elif isinstance(data, tuple):
        lines.append(f"{'  ' * indent}{prefix}: tuple")
        for index, item in enumerate(data):
            new_tuple_prefix = f"{prefix}[{index}]"
            lines.append(
                f"{'  ' * (indent + 1)}{new_tuple_prefix}: {type(item).__name__}")
            if isinstance(item, (Mapping, list, tuple)):
                lines.extend(print_dict_types(
                    item, new_tuple_prefix, indent + 2))
    else:
        lines.append(f"{'  ' * indent}{prefix}: {type(data).__name__}")

    if indent == 0:
        for line in lines:
            print(line)

    return lines
