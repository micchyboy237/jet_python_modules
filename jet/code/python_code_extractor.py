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


def strip_comments(content: str) -> str:
    """
    Remove comments outside of triple-quoted strings.
    Preserves entire triple-quoted strings including any inline `#` comments within.
    """
    triple_quote_pattern = re.compile(r"('''|\"\"\")")
    lines = content.splitlines()
    result_lines = []
    in_triple_quote = False
    current_quote = ""

    for line in lines:
        if not in_triple_quote:
            match = triple_quote_pattern.search(line)
            if match:
                current_quote = match.group(1)
                if line.count(current_quote) == 2:
                    # Opening and closing on the same line — preserve entire line
                    result_lines.append(line)
                    continue
                in_triple_quote = True
                result_lines.append(line)
            else:
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue  # remove full-line comment
                elif '#' in line:
                    code = line.split('#', 1)[0].rstrip()
                    if code:
                        result_lines.append(code)
                else:
                    result_lines.append(line)
        else:
            result_lines.append(line)
            if current_quote in line:
                if line.count(current_quote) % 2 == 1:
                    in_triple_quote = False

    # Clean up multiple blank lines
    cleaned = re.sub(r'\n\s*\n', '\n', '\n'.join(result_lines)).strip()
    return cleaned


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
