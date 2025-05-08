import os
import subprocess
from typing import Generator, List


def run_command(command: str, *, work_dir: str = None, separator: str = " ") -> Generator[str, None, None]:
    """
    Execute a command and yield its output line by line.

    :param command: Command string to execute
    :param work_dir: Directory to execute the command in
    :param separator: String used to split the command into parts (default: " ")
    :return: Generator yielding command output lines
    """
    if not command:
        raise ValueError("Command cannot be empty")

    try:
        if work_dir:  # Expand and normalize work_dir
            work_dir = os.path.expanduser(work_dir)
            work_dir = os.path.abspath(work_dir)

        cmd_parts = command.split(separator)
        process = subprocess.Popen(
            cmd_parts, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        for line in process.stdout:
            message = line.decode('utf-8').strip()
            data = message

            if data.startswith(("data: ", "result: ", "error: ")):
                yield f"{data}\n\n"
            else:
                yield f"other: {data}\n\n"
        for line in process.stderr:
            yield f"error: {line.decode('utf-8').strip()}\n\n"
    except Exception as e:
        raise RuntimeError(f"Error executing command: {e}")


def run_commands(commands: List[str], work_dir: str = None) -> Generator[str, None, None]:
    """
    Execute a list of commands and yield their output sequentially.

    :param commands: List of command strings to execute
    :param work_dir: Directory to execute the commands in
    :return: Generator yielding command output lines
    """
    for command in commands:
        yield f"command: {command}\n"
        yield from run_command(command, work_dir)
        yield "\n"
