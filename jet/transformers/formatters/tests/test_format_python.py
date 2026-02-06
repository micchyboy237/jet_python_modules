from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from black import InvalidInput
from jet.transformers.formatters import format_python


@pytest.fixture
def temp_py_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "example.py"
    file_path.write_text(
        dedent("""\
            def hello():print("Hello");x=1
            class A:pass
        """),
        encoding="utf-8",
    )
    return file_path


def test_formats_string_correctly():
    # Given
    ugly_code = dedent("""\
        def f(x,y):return x+y
        if True:print("hi")
    """)
    expected = dedent("""\
        def f(x, y):
            return x + y


        if True:
            print("hi")
    """)

    # When
    result = format_python(ugly_code)

    # Then
    assert result == expected


def test_formats_existing_file(temp_py_file: Path):
    # Given
    expected_snippet = 'def hello():\n    print("Hello")\n    x = 1\n'

    # When
    result = format_python(temp_py_file)

    # Then
    assert expected_snippet in result
    assert "class A:" in result  # minimal check that second block is there


def test_treats_non_existing_path_as_string_content():
    # Given
    fake_path = Path("print('hello'); x=1;y=2")

    # When
    result = format_python(fake_path)  # fake_path as Path

    # Then
    expected = dedent("""\
        print("hello")
        x = 1
        y = 2
    """)
    assert result == expected


def test_raises_on_invalid_syntax():
    # Given
    invalid = dedent("""\
        def f(:
            pass
    """)

    # When / Then
    with pytest.raises(InvalidInput, match="Cannot parse"):
        format_python(invalid)


def test_handles_empty_input():
    # Given
    empty = ""

    # When
    result = format_python(empty)

    # Then
    assert result == ""


def test_raises_when_path_is_directory(tmp_path: Path):
    # Given
    dir_path = tmp_path / "mydir"
    dir_path.mkdir()

    # When / Then
    with pytest.raises(IsADirectoryError, match="Expected a file"):
        format_python(dir_path)
