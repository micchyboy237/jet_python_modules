from typing import Any, List, Optional
from collections.abc import Mapping


def _are_dicts_same_structure(dict1: Mapping, dict2: Mapping) -> bool:
    """Check if two dictionaries have the same keys and value types, including nested structures."""
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        val1, val2 = dict1[key], dict2[key]
        if type(val1).__name__ != type(val2).__name__:
            return False
        if isinstance(val1, Mapping):
            if not _are_dicts_same_structure(val1, val2):
                return False
        elif isinstance(val1, list):
            if not isinstance(val2, list) or len(val1) != len(val2):
                return False
            if val1 and val2:
                # Check if all elements in both lists have the same types
                if any(type(v1).__name__ != type(v2).__name__ for v1, v2 in zip(val1, val2)):
                    return False
                # Recursively check nested dictionaries in lists
                if any(isinstance(v1, Mapping) and not _are_dicts4_same_structure(v1, v2)
                       for v1, v2 in zip(val1, val2) if isinstance(v1, Mapping)):
                    return False
    return True


def _get_common_dict_structure(data: List[Mapping]) -> Optional[Mapping]:
    """Return a representative dictionary if all items in the list have the same structure."""
    if not data or not all(isinstance(item, Mapping) for item in data):
        return None
    first_dict = data[0]
    if all(_are_dicts_same_structure(first_dict, item) for item in data[1:]):
        return first_dict
    return None


def print_dict_types(data: Any, prefix: str = "", indent: int = 0) -> List[str]:
    """
    Returns a list of strings describing the type of each value in a dictionary with full key paths,
    handling nested dictionaries and lists. Merges list items with identical dictionary structures
    into a single representation. Only uses numerical indices for tuples, not lists.

    Args:
        data: The input data to analyze (typically a dictionary)
        prefix: The current key path prefix (used in recursion)
        indent: The current indentation level for formatting

    Returns:
        List of strings, each representing a key path and its value type
    """
    lines = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, list):
                common_dict = _get_common_dict_structure(
                    value) if value else None
                if common_dict:
                    lines.append(f"{'  ' * indent}{new_prefix}[]: list[dict]")
                    lines.extend(print_dict_types(
                        common_dict, f"{new_prefix}[]", indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{new_prefix}: list")
                    # Handle empty lists without recursion
                    if not value:
                        continue
                    for item in value:
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
        common_dict = _get_common_dict_structure(data) if data else None
        if common_dict:
            lines.append(f"{'  ' * indent}{prefix}[]: list[dict]")
            lines.extend(print_dict_types(
                common_dict, f"{prefix}[]", indent + 1))
        else:
            lines.append(f"{'  ' * indent}{prefix}: list")
            # Handle empty lists without recursion
            if not data:
                return lines
            for item in data:
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

    for line in lines:
        print(line)

    return lines
