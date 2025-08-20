import ast
from textwrap import dedent


def get_signature(node, content, indent=0):
    source = ast.get_source_segment(content, node).splitlines()
    signature_lines = []
    for line in source:
        stripped = line.rstrip()
        signature_lines.append(stripped)
        if stripped.endswith(":"):
            break
    # Remove trailing colon for function definitions, but keep for classes
    if signature_lines and signature_lines[-1].endswith(":") and not isinstance(node, ast.ClassDef):
        signature_lines[-1] = signature_lines[-1][:-1]
    return "\n".join("    " * indent + line for line in signature_lines)


def shorten_functions(content: str, remove_class_vars: bool = False) -> str:
    """
    Shorten function and class definitions to their signatures, optionally removing class variables.
    Args:
        content: The source code to process.
        remove_class_vars: If True, exclude class-level variables in the output; if False, include them.
    Returns:
        A string containing only the signatures of functions and classes.
    """
    tree = ast.parse(content)
    definitions = []

    def process_node(node, indent=0):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(node, 'parent') or not isinstance(node.parent, ast.ClassDef):
                definitions.append(get_signature(node, content, indent))
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                child.parent = node
            class_lines = [get_signature(node, content, indent)]
            if not remove_class_vars:
                for body_node in node.body:
                    if isinstance(body_node, ast.AnnAssign):
                        var_line = ast.get_source_segment(
                            content, body_node).rstrip()
                        class_lines.append("    " * (indent + 1) + var_line)
                    elif isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_lines.append(get_signature(
                            body_node, content, indent=indent + 1))
                    elif isinstance(body_node, ast.ClassDef):
                        for child in ast.iter_child_nodes(body_node):
                            child.parent = body_node
                        class_lines.append(process_node(
                            body_node, indent=indent + 1))
            else:
                for body_node in node.body:
                    if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_lines.append(get_signature(
                            body_node, content, indent=indent + 1))
                    elif isinstance(body_node, ast.ClassDef):
                        for child in ast.iter_child_nodes(body_node):
                            child.parent = body_node
                        class_lines.append(process_node(
                            body_node, indent=indent + 1))
            return "\n".join(class_lines)

    for node in tree.body:  # Only process top-level nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result = process_node(node)
            if result:
                definitions.append(result)

    return dedent("\n".join(definitions))
