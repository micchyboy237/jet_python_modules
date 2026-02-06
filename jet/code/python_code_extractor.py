import re
from pathlib import Path

from jet.logger import logger


def preprocess_content(content: str):
    """Comment out lines starting with '!' or '%'."""
    content_lines = content.splitlines()
    processed_lines = [
        f"# {line}" if line.lstrip().startswith(("!", "%")) else line
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
                    if not (
                        remove_triple_quoted_definitions and current_quote == '"""'
                    ):
                        result_lines.append(line)
                    continue
                in_triple_quote = True
                if not (remove_triple_quoted_definitions and current_quote == '"""'):
                    result_lines.append(line)
            else:
                stripped = line.strip()
                if stripped.startswith("#"):
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
                    elif ch == "#" and not in_single and not in_double:
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

    cleaned = re.sub(r"\n\s*\n", "\n", "\n".join(result_lines)).strip()
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
            with open(source, encoding="utf-8") as file:
                content = file.read()
            modified_content = strip_comments(content, remove_triple_quoted_definitions)

            # Write the modified content back to the file
            with open(source, "w", encoding="utf-8") as file:
                file.write(modified_content)
            logger.log(
                "Removed comments from:", source, colors=["SUCCESS", "BRIGHT_SUCCESS"]
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


def extract_imports(source: str) -> str:
    """
    Extract every import statement (import / from ... import) from the code,
    including multiline ones using parentheses or backslash continuation.

    Args:
        source: The complete source code as a single string

    Returns:
        All import lines joined as a single string, preserving original formatting
        and line endings. Returns empty string if no imports are found.
    """
    if not source:
        return ""

    # Split preserving original line endings
    source_lines: list[str] = source.splitlines(keepends=True)

    imports: list[str] = []
    i = 0
    import_start_pattern = re.compile(r"^\s*(import\s|from\s)")

    print("[DEBUG:extract_imports] Starting scan")
    print(f"[DEBUG] Total lines: {len(source_lines)}")

    while i < len(source_lines):
        line = source_lines[i]
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if import_start_pattern.match(stripped):
            print(f"[DEBUG] Found import at line {i + 1}: {line.rstrip()}")
            block: list[str] = [line]
            open_parens = line.count("(") - line.count(")")
            ends_with_backslash = line.rstrip("\r\n").endswith("\\")

            i += 1
            while i < len(source_lines) and (open_parens > 0 or ends_with_backslash):
                cont_line = source_lines[i]
                print(f"[DEBUG]   → continuation line {i + 1}: {cont_line.rstrip()}")
                block.append(cont_line)
                open_parens += cont_line.count("(") - cont_line.count(")")
                ends_with_backslash = cont_line.rstrip("\r\n").endswith("\\")
                i += 1

            imports.extend(block)
            continue

        i += 1

    print(f"[DEBUG:extract_imports] Collected {len(imports)} import-related lines")

    if imports:
        print("[DEBUG] Last collected line:", repr(imports[-1]))

    # Since we used splitlines(keepends=True), just concatenate
    return "".join(imports).rstrip("\r\n") + "\n" if imports else ""


def move_imports_to_top(source: str) -> str:
    if not source.strip():
        return source

    imports_block = extract_imports(source)
    if not imports_block:
        return source

    # Get all lines with original endings
    lines = source.splitlines(keepends=True)

    # We'll mark lines that belong to imports
    import_line_indices = set()

    i = 0
    import_start_pattern = re.compile(r"^\s*(import\s|from\s)")

    while i < len(lines):
        stripped = lines[i].lstrip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        if import_start_pattern.match(stripped):
            # start of import block
            start = i
            open_parens = lines[i].count("(") - lines[i].count(")")
            ends_with_backslash = lines[i].rstrip("\r\n").endswith("\\")

            import_line_indices.add(i)
            i += 1

            while i < len(lines) and (open_parens > 0 or ends_with_backslash):
                import_line_indices.add(i)
                cont = lines[i]
                open_parens += cont.count("(") - cont.count(")")
                ends_with_backslash = cont.rstrip("\r\n").endswith("\\")
                i += 1
            continue

        i += 1

    # Rebuild non-import part
    non_import_lines = [
        line for idx, line in enumerate(lines) if idx not in import_line_indices
    ]

    # Combine: imports first, then rest
    # Add one blank line after imports if there's remaining code
    result_lines = []
    if imports_block:
        result_lines.append(imports_block)
        # Only add extra newline if there's actual code after
        if non_import_lines and not all(
            l.isspace() or l == "\n" or l == "\r\n" for l in non_import_lines
        ):
            result_lines.append("\n")

    result_lines.extend(non_import_lines)

    return "".join(result_lines).strip() + "\n"
