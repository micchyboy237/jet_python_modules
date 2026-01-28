import fnmatch
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from rich.table import Table
from jet.logger import logger
from jet.transformers.formatters import format_html
from jet.transformers.object import make_serializable

def get_file_last_modified(file_path: str) -> float:
    """Get the last modified time of a file."""
    return os.path.getmtime(file_path)


def load_json(file_path):
    """Load existing results from the JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def merge_json(existing_json, new_results):
    """
    Recursively merge two data structures.
    Handles both dictionaries and lists.
    """
    if isinstance(new_results, dict):
        for key, value in new_results.items():
            if key in existing_json and isinstance(existing_json[key], dict) and isinstance(value, dict):
                existing_json[key] = merge_json(existing_json[key], value)
            else:
                existing_json[key] = value
    elif isinstance(new_results, list):
        if not isinstance(existing_json, list):
            existing_json = []
        for i in range(min(len(existing_json), len(new_results))):
            existing_json[i] = merge_json(existing_json[i], new_results[i])
        if len(existing_json) < len(new_results):
            existing_json.extend(new_results[len(existing_json):])
    else:
        existing_json = new_results
    return existing_json


def save_json(results, file_path="generated/results.json"):
    """
    Save results to a JSON file, merging with existing results if the file exists.
    """
    try:
        # Serialize results
        results = make_serializable(results)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Load existing results
        existing_json = load_json(file_path)

        # Merge existing results with new results
        merged_results = merge_json(existing_json, results)

        # Save the merged results to the file
        with open(file_path, "w") as f:
            json.dump(merged_results, f, indent=2, sort_keys=True)

        # Log success message
        logger.log("Results saved to", file_path,
                   colors=["SUCCESS", "BRIGHT_SUCCESS"])

    except Exception as e:
        logger.log("Error saving results:", str(e), colors=["GRAY", "RED"])
        raise e


def load_file(input_file: str, verbose: bool = True) -> Any:
    if not os.path.exists(input_file):
        if verbose:
            logger.warning(f"File does not exist: {input_file}")
        return None

    try:
        ext = Path(input_file).suffix.lower()

        # JSON or JSONL
        if ext in {".json", ".jsonl"}:
            if input_file.endswith(".json"):
                with open(input_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        if verbose:
                            logger.error(f"Invalid JSON format in {input_file}: {e}")
                        return None
            else:  # .jsonl
                data = []
                with open(input_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                if verbose:
                                    logger.error(f"Invalid JSONL line in {input_file}: {e}")
                                return None

            if verbose:
                prefix = f"Loaded JSON{'L' if input_file.endswith('.jsonl') else ''} data {len(data) if isinstance(data, list) else ''} from: "
                logger.newline()
                logger.log(prefix, input_file, colors=["INFO", "BRIGHT_INFO"])
            return data

        # HTML or Text
        elif ext in {".html", ".txt"}:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
            if ext == ".html":
                content = format_html(content)
                if verbose:
                    logger.newline()
                    logger.log("Loaded and formatted HTML from: ", input_file, colors=["INFO", "BRIGHT_INFO"])
            else:
                if verbose:
                    logger.newline()
                    logger.log("Loaded text from: ", input_file, colors=["INFO", "BRIGHT_INFO"])
            return content

        # Binary files (e.g., PNG, JPEG)
        elif ext in {".png", ".jpg", ".jpeg"}:
            with open(input_file, "rb") as f:
                content = f.read()
            if verbose:
                logger.newline()
                logger.log("Loaded binary data from: ", input_file, colors=["INFO", "BRIGHT_INFO"])
            return content

        # Other files
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                content = f.read()
            if verbose:
                logger.newline()
                logger.log("Loaded data from: ", input_file, colors=["INFO", "BRIGHT_INFO"])
            return content

    except UnicodeDecodeError:
        # Fallback for binary files not explicitly handled
        with open(input_file, "rb") as f:
            content = f.read()
        if verbose:
            logger.newline()
            logger.log("Loaded binary data from: ", input_file, colors=["INFO", "BRIGHT_INFO"])
        return content
    except Exception as e:
        if verbose:
            logger.newline()
            logger.error(f"Failed to load file: {e}")
        raise


def save_file(
    data: Union[str, bytes, List, Dict, BaseModel, Table],
    output_file: Union[str, Path],
    verbose: bool = True,
    append: bool = False
) -> str:
    """
    Save data to a file in various formats based on extension.

    Args:
        data: The data to save (str, bytes, List, Dict, BaseModel, or Table).
        output_file: The path where the file will be saved.
        verbose: Whether to log save operations.
        append: Whether to append to JSONL files (ignored for other formats).

    Returns:
        The output file path as a string.

    Raises:
        ValueError: If data type is unsupported or invalid for the file format.
        Exception: If file operations fail.
    """
    from rich.console import Console
    from rich.padding import Padding
    from rich.text import Text
    from pathlib import Path

    output_path = Path(output_file).expanduser()
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)
    try:
        ext = Path(output_file).suffix.lower()
        if ext in {".json", ".jsonl"}:
            if isinstance(data, Table):
                if not data.columns or not data.rows:
                    raise ValueError("Table is empty or has no columns/rows")
                columns = [col.header for col in data.columns]
                table_data = []
                console = Console(record=True)
                for row_idx in range(len(data.rows)):
                    row_data = {}
                    for col_idx, col in enumerate(data.columns):
                        cells = list(data._get_cells(console, col_idx, col))
                        cell_content = cells[row_idx].renderable if row_idx < len(cells) else ""
                        # Extract inner renderable from Padding if present
                        if isinstance(cell_content, Padding):
                            cell_content = getattr(cell_content, 'renderable', cell_content)
                        row_data[columns[col_idx]] = str(cell_content)
                    table_data.append(row_data)
                data = make_serializable(table_data)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError as e:
                    if verbose:
                        logger.error(f"Invalid JSON string: {e}")
                    raise
            else:
                data = make_serializable(data)
            if ext == ".json":
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                mode = "a" if append and os.path.exists(output_file) else "w"
                with open(output_file, mode, encoding="utf-8") as f:
                    if isinstance(data, list):
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    else:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if verbose:
                logger.newline()
                upper_ext = ext.upper().lstrip('.')
                count = f" {len(data)}" if isinstance(data, list) else ""
                action = 'Appended' if append else 'Saved'
                path_text = Text(output_file, style="bold bright_green")
                path_text.overflow = "fold"
                path_text.no_wrap = True
                logger.log(f"{action} {upper_ext} data{count} to: ", path_text, colors=["SUCCESS", "BRIGHT_SUCCESS"])
        elif ext == ".md":
            if isinstance(data, Table):
                if not data.columns or not data.rows:
                    raise ValueError("Table is empty or has no columns/rows")
                columns = [str(col.header).replace("|", "\\|").replace("\n", " ") for col in data.columns]
                console = Console(record=True)
                table_data = []
                for row_idx in range(len(data.rows)):
                    row_data = {}
                    for col_idx, col in enumerate(data.columns):
                        cells = list(data._get_cells(console, col_idx, col))
                        cell_content = cells[row_idx].renderable if row_idx < len(cells) else ""
                        # Extract inner renderable from Padding if present
                        if isinstance(cell_content, Padding):
                            cell_content = getattr(cell_content, 'renderable', cell_content)
                        cell_content = str(cell_content).replace("|", "\\|").replace("\n", " ")
                        row_data[columns[col_idx]] = cell_content
                    table_data.append(row_data)
                header_row = '| ' + ' | '.join(columns) + ' |'
                separator_row = '| ' + ' | '.join(['---' for _ in columns]) + ' |'
                data_rows = [
                    '| ' + ' | '.join(str(row.get(col, '')).replace("|", "\\|").replace("\n", " ") for col in columns) + ' |'
                    for row in table_data
                ]
                data = '\n'.join([header_row, separator_row] + data_rows)
            elif not isinstance(data, str):
                data = str(data).replace("|", "\\|").replace("\n", " ")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            if verbose:
                logger.newline()
                path_text = Text(output_file, style="bold bright_green")
                path_text.overflow = "fold"
                path_text.no_wrap = True
                logger.log("Saved Markdown to: ", path_text, colors=["SUCCESS", "BRIGHT_SUCCESS"])
        elif ext in {".html", ".txt"}:
            if not isinstance(data, str):
                data = str(data)
            if ext == ".html":
                data = format_html(data)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            if verbose:
                logger.newline()
                kind = 'HTML' if ext == '.html' else 'text'
                path_text = Text(output_file, style="bold bright_green")
                path_text.overflow = "fold"
                path_text.no_wrap = True
                logger.log(f"Saved {kind} to: ", path_text, colors=["SUCCESS", "BRIGHT_SUCCESS"])
        elif ext in {".png", ".jpg", ".jpeg"}:
            if not isinstance(data, bytes):
                raise ValueError(f"Expected bytes for {ext} file, got {type(data)}")
            with open(output_file, "wb") as f:
                f.write(data)
            if verbose:
                logger.newline()
                path_text = Text(output_file, style="bold bright_green")
                path_text.overflow = "fold"
                path_text.no_wrap = True
                logger.log("Saved binary data to: ", path_text, colors=["SUCCESS", "BRIGHT_SUCCESS"])
        else:
            if isinstance(data, bytes):
                with open(output_file, "wb") as f:
                    f.write(data)
            else:
                if not isinstance(data, str):
                    data = str(data)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(data)
            if verbose:
                logger.newline()
                path_text = Text(output_file, style="bold bright_green")
                path_text.overflow = "fold"
                path_text.no_wrap = True
                logger.log("Saved data to: ", path_text, colors=["SUCCESS", "BRIGHT_SUCCESS"])
        return output_file
    except Exception as e:
        if verbose:
            logger.newline()
            logger.error(f"Failed to save file: {e}")
        raise


def main():
    # Define file path
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path1 = os.path.join(file_dir, "generated/results1.json")
    file_path2 = os.path.join(file_dir, "generated/results2.json")
    """Example usage of saving results to JSON."""
    sample_results_1 = {
        "session_data": [
            "item1",
            2,
            {
                "item3": {
                    "task1": {"status": "completed", "details": {"time": "2 hours"}},
                    "task2": {"status": "pending", "priority": "high"},
                },
            }
        ]
    }
    sample_results_2 = [
        {
            "user_data": {
                "id": 1,
                "name": "John Doe",
                "preferences": {
                    "theme": True,
                    "notifications": "dark"
                }
            },
        }
    ]
    save_json(sample_results_1, file_path=file_path1)
    save_json(sample_results_2, file_path=file_path2)


def load_data(file_path: str, is_binary=False):
    has_no_extension = not os.path.splitext(file_path)[1]
    if has_no_extension or file_path.endswith(".bin"):
        is_binary = True

    if is_binary:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path).to_dict(orient='records')
    elif file_path.endswith(".txt") or file_path.endswith(".md"):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
    elif not os.path.isdir(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    else:
        data = load_data_from_directories([file_path])

    return data


def load_data_from_directories(source_directories, includes=None, excludes=None):
    data = []

    for directory in source_directories:
        # Check if directory is a json file
        if os.path.isfile(directory) and directory.endswith(".json"):
            source_file = directory
            with open(source_file, 'r') as file:
                data.extend(json.load(file))
            continue
        for filename in os.listdir(directory):
            # Apply include and exclude filters
            if (not includes or any(fnmatch.fnmatch(filename, pattern) for pattern in includes)) and \
               (not excludes or not any(fnmatch.fnmatch(filename, pattern) for pattern in excludes)):
                source_file = os.path.join(directory, filename)
                data.extend(load_data(source_file))

    return data


def save_data(output_file, data, overwrite=False, key='id', is_binary=False):
    if not data:
        print(f"No data to save for {output_file}")
        return
    # Check if the output file has no extension
    has_no_extension = not os.path.splitext(output_file)[1]
    if has_no_extension or output_file.endswith(".bin"):
        is_binary = True

    if overwrite or not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # data = [dict(t) for t in {tuple(d.items()) for d in data}]

        logger.success(f"Writing {len(data)} items to {output_file}")

        if is_binary:
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # Update existing data array with matching data array based on item['key]
        updated_data_dict = {
            item[key]: item for item in existing_data if key in item}
        for idx, item in enumerate(data):
            if item.get(key, None) in updated_data_dict:
                existing_data_index = next(
                    (i for i, x in enumerate(existing_data) if x[key] == item[key]), None)
                existing_data[existing_data_index] = {
                    **existing_data[existing_data_index],
                    **item
                }
            else:
                existing_data.append(item)

        # Deduplicate by key
        # existing_data = [dict(t) for t in {tuple(d.items()) for d in existing_data}]
        logger.success(f"Writing {len(existing_data)} items to {output_file}")

        if is_binary:
            with open(output_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
