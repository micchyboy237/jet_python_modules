from typing import TypedDict


class CodeBlock(TypedDict):
    """Represents a code block with associated metadata."""
    type: str  # Programming language of the code block.
    code: str  # The code content.


def rst_to_code_blocks(rst_file_path: str) -> list[CodeBlock]:
    """
    Extracts all code blocks from an .rst file, including their types and content.

    Args:
    - rst_file_path (str): Path to the input reStructuredText (.rst) file.

    Returns:
    - list[dict]: A list of dictionaries containing 'type' (language type) and 'code' (code content).
    """
    from docutils import core

    # Read the contents of the .rst file
    with open(rst_file_path, 'r') as rst_file:
        rst_content = rst_file.read()

    # Parse the reStructuredText content
    document = core.publish_doctree(rst_content)

    # Extract all code blocks
    code_blocks = []
    for node in document.traverse():
        if node.tagname == 'literal_block':
            # Extract the type (language) and the code content
            # Default to 'text' if no class is specified
            code_type = node.attributes.get('classes', ['text'])[-1]
            code_content = node.astext()
            code_blocks.append({'type': code_type, 'code': code_content})

    return code_blocks
