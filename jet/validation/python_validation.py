from typing import Optional, List, TypedDict, Union
import ast
import os
from jet.logger import logger


class CodeValidationError(TypedDict):
    message: str
    code: Optional[str]
    file: Optional[str]
    line: Optional[int]


def get_code_as_string(source: str) -> str:
    """
    Convert a file or code string to a string of Python code.
    """
    if os.path.exists(source):
        with open(source, 'r') as file:
            return file.read()
    return source


def check_syntax_error_for_source(source: str) -> Optional[List[CodeValidationError]]:
    """
    Check syntax errors for a single source (string or file path).
    """
    code = get_code_as_string(source)
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return [CodeValidationError(
            message=f"SyntaxError: {e.msg}",
            file=os.path.realpath(source) if os.path.exists(source) else None,
            line=e.lineno,
            code=code.splitlines()[e.lineno - 1] if e.lineno else None
        )]


def validate_python_syntax(code_sources: Union[str, List[str]]) -> Optional[List[CodeValidationError]]:
    """
    Validate syntax across one or more Python sources.
    """
    errors: List[CodeValidationError] = []

    if isinstance(code_sources, list):
        for src in code_sources:
            result = check_syntax_error_for_source(src)
            if result:
                errors.extend(result)
    else:
        result = check_syntax_error_for_source(code_sources)
        if result:
            errors.extend(result)

    return errors if errors else None
