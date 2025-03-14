import fnmatch
import os
import json
import pickle
import pandas as pd

from typing import Optional
from jet.logger import logger
from pydantic.main import BaseModel


def get_file_last_modified(file_path: str) -> float:
    """Get the last modified time of a file."""
    return os.path.getmtime(file_path)


def load_json(file_path):
    """Load existing results from the JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
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
    from jet.transformers import make_serializable
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
        logger.log(f"Error saving results:", str(e), colors=["GRAY", "RED"])
        raise e


def load_file(input_file: str) -> Optional[str | dict | list]:
    import os
    import json
    from jet.logger import logger  # Ensure this is your logger module

    # Check if file exists
    if not os.path.exists(input_file):
        logger.warning(f"File does not exist: {input_file}")
        # raise FileNotFoundError(f"File not found: {input_file}")
        return None

    try:
        if input_file.endswith(".json"):
            with open(input_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except:
                    data = None

            if isinstance(data, list):
                prefix = f"Loaded JSON data {len(data)} from:"
            else:
                prefix = "Loaded JSON data from:"

            logger.log(
                prefix,
                input_file,
                colors=["INFO", "BRIGHT_INFO"]
            )
            return data
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                data = f.read()
            logger.log(
                "Loaded data from:",
                input_file,
                colors=["INFO", "BRIGHT_INFO"]
            )
            return data
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        raise


def save_file(data: str | dict | list | BaseModel, output_file: str):
    import os
    from jet.transformers import make_serializable
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to file
    try:
        if output_file.endswith(".json"):
            if isinstance(data, str):
                data = json.loads(data)
            else:
                data = make_serializable(data)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            if isinstance(data, list):
                prefix = f"Save JSON data {len(data)} to:"
            else:
                prefix = "Save JSON data to:"

            logger.log(
                prefix,
                output_file,
                colors=["SUCCESS", "BRIGHT_SUCCESS"]
            )
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(data)
            logger.log(
                "Save data to:",
                output_file,
                colors=["SUCCESS", "BRIGHT_SUCCESS"]
            )
    except Exception as e:
        logger.error(f"Failed to save file: {e}")


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


def save_data(output_file, data, write=False, key='id', is_binary=False):
    if not data:
        print(f"No data to save for {output_file}")
        return
    # Check if the output file has no extension
    has_no_extension = not os.path.splitext(output_file)[1]
    if has_no_extension or output_file.endswith(".bin"):
        is_binary = True

    if write or not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # data = [dict(t) for t in {tuple(d.items()) for d in data}]

        print(f"Writing {len(data)} items to {output_file}")

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
        print(f"Writing {len(existing_data)} items to {output_file}")

        if is_binary:
            with open(output_file, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
