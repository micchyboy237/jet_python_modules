import black
from pathlib import Path


def format_python_code(source_code: str) -> str:
    """
    Formats Python code using the Black formatter.

    Args:
        source_code (str): The source code to be formatted.

    Returns:
        str: The formatted Python code.
    """
    try:
        # Specify Black formatting mode
        mode = black.Mode()
        # Format code
        formatted_code = black.format_str(source_code, mode=mode)
        return formatted_code
    except black.InvalidInput as e:
        raise ValueError(f"Invalid Python code provided: {e}")


def format_file(file_path: str) -> None:
    """
    Formats a Python file in place using Black.

    Args:
        file_path (str): Path to the Python file to format.
    """
    try:
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Read the content of the file
        source_code = file.read_text()
        # Format the content
        formatted_code = format_python_code(source_code)
        # Write the formatted content back to the file
        file.write_text(formatted_code)
        print(f"File '{file_path}' formatted successfully.")
    except Exception as e:
        print(f"Error: {e}")


# Example Usage
if __name__ == "__main__":
    # Example 1: Format a string
    unformatted_code = "def add(a,b):return a+b"
    print("Formatted Code:")
    print(format_python_code(unformatted_code))

    # Example 2: Format a Python file
    # Uncomment the following line and replace with your file path to test
    # format_file("path/to/your_file.py")
