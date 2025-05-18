import ast
import os
from typing import Dict, List, Tuple
import re


def extract_class_and_function_defs(target_dir: str) -> Dict[str, List[str]]:
    def_nodes = {}
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    lines = source.splitlines()
                    tree = ast.parse(source, filename=full_path)
                    extracted: List[str] = []

                    def process_node(node: ast.AST):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            header = lines[node.lineno - 1].rstrip()
                            doc_lines = []
                            if node.body:
                                first_stmt = node.body[0]
                                if (
                                    isinstance(first_stmt, ast.Expr)
                                    and isinstance(first_stmt.value, ast.Constant)
                                    and isinstance(first_stmt.value.value, str)
                                ):
                                    doc_lines = lines[first_stmt.lineno -
                                                      1: first_stmt.end_lineno]
                            extracted.append("\n".join([header] + doc_lines))
                        elif isinstance(node, ast.ClassDef):
                            header = lines[node.lineno - 1].rstrip()
                            doc_lines = []
                            if node.body:
                                first_stmt = node.body[0]
                                if (
                                    isinstance(first_stmt, ast.Expr)
                                    and isinstance(first_stmt.value, ast.Constant)
                                    and isinstance(first_stmt.value.value, str)
                                ):
                                    doc_lines = lines[first_stmt.lineno -
                                                      1: first_stmt.end_lineno]
                            extracted.append("\n".join([header] + doc_lines))
                            for child in node.body:
                                process_node(child)

                    for node in ast.iter_child_nodes(tree):
                        process_node(node)

                    if extracted:
                        def_nodes[full_path] = extracted
                except (SyntaxError, UnicodeDecodeError) as e:
                    print(f"Error processing {full_path}: {e}")

    # Helper function to sort with numeric prefix if any
    def sort_key(path: str) -> Tuple[int, str]:
        filename = os.path.basename(path)
        # Extract leading number if any
        match = re.match(r"(\d+)(.*)", filename)
        if match:
            num_part = int(match.group(1))
            rest = match.group(2)
            return (num_part, rest)
        # Files without leading number come after numbers
        return (float('inf'), filename)

    # Create a new dict sorted by the key function
    sorted_def_nodes = dict(
        sorted(def_nodes.items(), key=lambda kv: sort_key(kv[0])))

    return sorted_def_nodes
