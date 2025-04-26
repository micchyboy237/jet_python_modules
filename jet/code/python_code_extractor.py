import ast
import re
from pathlib import Path

from jet.logger import logger


def preprocess_content(content: str):
    """Comment out lines starting with '!' or '%'."""
    content_lines = content.splitlines()
    processed_lines = [
        f"# {line}" if line.lstrip().startswith(('!', '%')) else line
        for line in content_lines
    ]
    processed_content = "\n".join(processed_lines)
    return processed_content


def strip_comments(content):
    """
    Remove all comments except lines starting with '#' that are inside triple-quoted strings.
    """
    content = preprocess_content(content)
    lines = content.split('\n')
    triple_quoted_lines = set()

    # Find all triple-quoted blocks (both ''' and """)
    triple_quote_pattern = re.compile(r'([\'"]{3})')
    in_triple_quote = False
    quote_type = ''
    start_line = 0

    for idx, line in enumerate(lines):
        if not in_triple_quote:
            match = triple_quote_pattern.search(line)
            if match:
                quote_type = match.group(1)
                if line.count(quote_type) == 2:
                    # Opening and closing on same line — skip it
                    continue
                in_triple_quote = True
                start_line = idx
        else:
            if quote_type in line:
                for i in range(start_line, idx + 1):
                    triple_quoted_lines.add(i)
                in_triple_quote = False

    result_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i in triple_quoted_lines:
            if stripped.startswith('#'):
                # Preserve comment inside triple-quoted block
                result_lines.append(line)
            continue  # Skip everything else in triple-quoted strings
        elif stripped.startswith('#'):
            continue  # Remove regular comment
        elif '#' in line:
            code_part = line.split('#')[0].rstrip()
            if code_part:
                result_lines.append(code_part)
        else:
            result_lines.append(line)

    content = '\n'.join(result_lines)
    return re.sub(r'\n\s*\n', '\n', content).strip()


def remove_comments(source):
    """
    Remove comments from either a Python file or Python code string, retaining lines starting with '#'.

    Args:
        source (str): Path to the Python file or Python code string.

    Returns:
        str: For files: Success or error message.
             For code string: Modified code with comments removed.
    """
    # Check if source is a file path
    if Path(source).is_file():
        try:
            # Read the file content
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()

            # Process the content
            modified_content = strip_comments(content)

            # Write the modified content back to the file
            with open(source, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            logger.log(
                "Removed comments from:",
                source,
                colors=["SUCCESS", "BRIGHT_SUCCESS"]
            )

            return

        except FileNotFoundError:
            return f"Error: File '{source}' not found"
        except SyntaxError:
            return "Error: Invalid Python syntax in the file"
        except Exception as e:
            return f"Error: {str(e)}"

    # Treat source as Python code string
    else:
        try:
            # Process the content
            modified_content = strip_comments(source)
            return modified_content

        except SyntaxError:
            return "Error: Invalid Python syntax in the code"
        except Exception as e:
            return f"Error: {str(e)}"
