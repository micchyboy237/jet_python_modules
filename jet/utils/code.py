import ast


def shorten_functions(content):
    """Extracts function and class definitions from Python code, including full signatures but excluding bodies."""
    tree = ast.parse(content)
    content_lines = content.splitlines()
    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno - 1
            end_line = node.body[0].lineno - 1  # Line before the body starts

            # Collect all lines of the signature
            signature_lines = content_lines[start_line:end_line]
            signature = "\n".join(line.rstrip() for line in signature_lines)
            definitions.append(signature)

    return "\n".join(definitions)
