# test_move_imports_to_top.py
from textwrap import dedent

import pytest
from jet.code.python_code_extractor import move_imports_to_top


@pytest.mark.parametrize(
    "name, before, expected",
    [
        # 1. Basic case
        (
            "basic imports",
            dedent("""\
                print("hello")
                import os
                from pathlib import Path
                import sys as system
                x = 10
            """),
            dedent("""\
                import os
                from pathlib import Path
                import sys as system

                print("hello")
                x = 10
            """),
        ),
        # 2. Multiline import (parentheses)
        (
            "multiline parentheses",
            dedent("""\
                def hello():
                    pass

                from typing import (
                    Any,
                    Dict,
                    List,
                    Optional,
                )

                class Example:
                    pass
            """),
            dedent("""\
                from typing import (
                    Any,
                    Dict,
                    List,
                    Optional,
                )


                def hello():
                    pass

                class Example:
                    pass
            """),
        ),
        # 3. Imports inside functions / classes / if blocks
        (
            "late / conditional imports",
            dedent("""\
                def func():
                    import numpy
                    from datetime import datetime
                    return datetime.now()

                if True:
                    import math

                print("done")
            """),
            dedent("""\
                import numpy
                from datetime import datetime
                import math

                def func():
                    return datetime.now()

                if True:
                    pass

                print("done")
            """),
        ),
        # 4. Try/except imports + mixed
        (
            "try/except imports",
            dedent("""\
                try:
                    import ujson as json
                except ImportError:
                    import json

                import os

                print("ok")
            """),
            dedent("""\
                try:
                    import ujson as json
                except ImportError:
                    import json

                import os

                print("ok")
            """),
        ),
        # 5. Backslash continuation (rare but valid)
        (
            "backslash continuation",
            dedent("""\
                from some.very.long.package.name.that.goes.on \\
                    import SomethingVeryImportant

                x = 1
            """),
            dedent("""\
                from some.very.long.package.name.that.goes.on \\
                    import SomethingVeryImportant

                x = 1
            """),
        ),
        # 6. File starting with blank lines / docstring
        (
            "preserve leading blank lines & docstring",
            dedent("""\


                \"\"\"Module docstring\"\"\"

                import re
                import sys

                def main():
                    pass
            """),
            dedent("""\


                \"\"\"Module docstring\"\"\"

                import re
                import sys

                def main():
                    pass
            """),
        ),
        # 7. No imports at all
        (
            "no imports",
            dedent("""\
                print(123)
                x = [1,2,3]
            """),
            dedent("""\
                print(123)
                x = [1,2,3]
            """),
        ),
    ],
    ids=lambda x: x,
)
def test_move_imports_to_top(name, before, expected):
    # Given
    source = before

    # When
    result = move_imports_to_top(source)

    # Then
    assert result == expected, f"Failed for case: {name}"
