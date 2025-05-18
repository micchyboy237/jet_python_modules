import tokenize
from io import StringIO
from typing import List, Tuple


def extract_comments(code: str) -> List[Tuple[str, int]]:
    """Extract comments from Python code, returning a list of (comment, line_number) tuples."""
    comments = []
    try:
        code_io = StringIO(code)
        tokens = tokenize.generate_tokens(code_io.readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_text = token.string.strip()
                if comment_text.startswith('#'):
                    comment_text = comment_text[1:].strip()
                comments.append((comment_text, token.start[0]))
    except tokenize.TokenError:
        # Return empty list for malformed code
        return comments
    return comments


def remove_comments(code: str) -> str:
    """Remove comments from Python code, preserving triple-quoted strings and whitespace."""
    try:
        code_io = StringIO(code)
        tokens = list(tokenize.generate_tokens(code_io.readline))
        # Filter out comment tokens and dedent tokens
        filtered_tokens = []
        for token in tokens:
            if token.type != tokenize.COMMENT:
                filtered_tokens.append(token)
        # Use untokenize to reconstruct the code
        result = tokenize.untokenize(filtered_tokens)
        # Normalize whitespace: strip trailing spaces per line and normalize newlines
        lines = [line.rstrip() for line in result.splitlines()]
        # Remove empty lines that were solely comments
        cleaned_lines = [line for line in lines if line.strip()]
        return '\n'.join(cleaned_lines).rstrip()
    except tokenize.TokenError:
        return code  # Return original code if parsing fails
