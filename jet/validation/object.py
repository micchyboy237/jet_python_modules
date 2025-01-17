from collections.abc import Iterable
import collections.abc
from typing import Any, Literal, Optional, Sequence, Type


def is_iterable_but_not_primitive(
        obj: Any, type: Optional[Literal["list", 'dict', 'set']]) -> bool:
    if isinstance(obj, (int, float, bool, str, bytes)):
        return False

    if type in ["list", 'dict', 'set']:
        if type == 'list' and isinstance(obj, (collections.abc.Sequence, collections.abc.MutableSequence, )):
            return True
        elif type == 'dict' and isinstance(obj, (collections.abc.Mapping, collections.abc.MutableMapping)):
            return True
        elif type == 'set' and isinstance(obj, (collections.abc.Set, collections.abc.MutableSet)):
            return True

    else:
        raise ValueError(f"Error on is_iterable_but_not_primitive:\n'type' arg must be {
                         ["list", 'dict', 'set']}")


__all__ = [
    "is_iterable_but_not_primitive",
]


# Ensure the main function runs when executed
if __name__ == "__main__":
    # Real-world usage examples with assertions
    test_cases = [
        ([1, 2, 3], True),        # list (collection)
        ((1, 2, 3), True),        # tuple (collection)
        ({1, 2, 3}, True),        # set (collection)
        ({'a': 1}, True),         # dict (collection)
        (123, False),             # int (primitive)
        (45.67, False),           # float (primitive)
        (True, False),            # bool (primitive)
        ('Hello', False),         # str (primitive)
        (b'Hello', False),        # bytes (primitive)
        (range(5), True),         # range (collection)
        (None, False),            # None (not iterable, primitive)
        (object(), False),        # generic object (not iterable)
    ]

    for obj, expected in test_cases:
        result = is_iterable_but_not_primitive(obj)
        print(f'Object: {obj}, Expected: {expected}, Result: {result}')
        assert result == expected, f"Assertion failed for {obj}"

    print("All assertions passed!")
