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


def strip_comments(content: str, remove_triple_quoted_definitions: bool = False) -> str:
    """
    Remove comments outside of triple-quoted strings and string literals.
    Preserves entire triple-quoted strings and inline '#' inside quotes.
    If remove_triple_quoted_definitions=True, removes all triple double quoted block definitions.
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
                    # Opening and closing on the same line
                    if not (remove_triple_quoted_definitions and current_quote == '"""'):
                        result_lines.append(line)
                    continue
                in_triple_quote = True
                if not (remove_triple_quoted_definitions and current_quote == '"""'):
                    result_lines.append(line)
            else:
                stripped = line.strip()
                if stripped.startswith('#'):
                    continue  # remove full-line comment

                # walk through chars and detect # only if not inside quotes
                new_line = []
                in_single = in_double = False
                i = 0
                while i < len(line):
                    ch = line[i]
                    if ch == "'" and not in_double:
                        in_single = not in_single
                        new_line.append(ch)
                    elif ch == '"' and not in_single:
                        in_double = not in_double
                        new_line.append(ch)
                    elif ch == '#' and not in_single and not in_double:
                        break  # start of comment outside quotes
                    else:
                        new_line.append(ch)
                    i += 1

                cleaned = "".join(new_line).rstrip()
                if cleaned:
                    result_lines.append(cleaned)
        else:
            # inside triple quotes
            if not (remove_triple_quoted_definitions and current_quote == '"""'):
                result_lines.append(line)

            if current_quote in line:
                if line.count(current_quote) % 2 == 1:
                    in_triple_quote = False

    cleaned = re.sub(r'\n\s*\n', '\n', '\n'.join(result_lines)).strip()
    return cleaned


def remove_comments(source, remove_triple_quoted_definitions: bool = False):
    """
    Remove comments from a Python file or code string.

    Args:
        source (str): Path to a Python file or a string of Python code.
        remove_triple_quoted_definitions (bool): If True, remove triple-quoted docstrings.

    Returns:
        str: If source is a file path, returns None after modifying the file in-place.
             If source is a code string, returns the code with comments removed.
    """
    # If source is a file path, process file in-place
    if Path(source).is_file():
        try:
            with open(source, 'r', encoding='utf-8') as file:
                content = file.read()
            modified_content = strip_comments(
                content, remove_triple_quoted_definitions)

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
