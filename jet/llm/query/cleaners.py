from typing import List, Dict
from llama_index.core.schema import BaseNode, TextNode


def group_and_merge_texts_by_file_name(nodes: List[BaseNode]) -> Dict[str, str]:
    """
    Groups and merges the texts of nodes by their metadata["file_name"] attribute,
    ensuring no overlapping or duplicate lines between consecutive nodes, and sorting
    by the start_line_idx to preserve correct order.

    Args:
        nodes (List[BaseNode]): A list of BaseNode or TextNode objects to group and merge.

    Returns:
        Dict[str, str]: A dictionary where keys are file names and values are the merged texts.
    """
    # Sort nodes by their start_line_idx
    nodes_sorted = sorted(
        nodes, key=lambda node: node.metadata.get("start_line_idx", 0))

    grouped_texts: Dict[str, List[str]] = {}

    for node in nodes_sorted:
        file_name = node.metadata.get("file_name", "unknown")
        if file_name not in grouped_texts:
            grouped_texts[file_name] = []

        node_lines = node.text.splitlines()
        if grouped_texts[file_name]:
            # Avoid duplicate lines by comparing the last line of existing text with the first line of new text
            last_existing_line = grouped_texts[file_name][-1]
            if node_lines[0] == last_existing_line:
                node_lines = node_lines[1:]

        grouped_texts[file_name].extend(node_lines)

    # Merge lines back into strings
    merged_texts = {file_name: "\n".join(
        lines) for file_name, lines in grouped_texts.items()}

    return merged_texts


# Example Usage
if __name__ == "__main__":
    nodes = [
        TextNode(text="Header 1\nContent 1",
                 metadata={"file_name": "file1.md"}),
        TextNode(text="Content 1\nHeader 2\nContent 2",
                 metadata={"file_name": "file1.md"}),
        TextNode(text="Header A\nContent A",
                 metadata={"file_name": "file2.md"}),
        TextNode(text="Content A\nHeader B\nContent B",
                 metadata={"file_name": "file2.md"}),
    ]

    result = group_and_merge_texts_by_file_name(nodes)

    for file_name, content in result.items():
        print(f"File: {file_name}\n{content}\n")
