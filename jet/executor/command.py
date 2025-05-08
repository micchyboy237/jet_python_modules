import os
import asyncio
import subprocess
from typing import Generator, AsyncGenerator, List


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
        if work_dir:
            work_dir = os.path.expanduser(work_dir)
            work_dir = os.path.abspath(work_dir)
        cmd_parts = command.split(separator)
        process = subprocess.Popen(
            cmd_parts,
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line-buffered
            universal_newlines=True  # Text mode
        )
        # Read stdout line by line in real-time
        while process.poll() is None:
            line = process.stdout.readline()
            if line:
                message = line.strip()
                if message.startswith(("data: ", "result: ", "error: ")):
                    yield f"{message}\n"
                else:
                    yield f"other: {message}\n"
        # Read any remaining stdout
        for line in process.stdout:
            message = line.strip()
            if message:
                if message.startswith(("data: ", "result: ", "error: ")):
                    yield f"{message}\n"
                else:
                    yield f"other: {message}\n"
        # Read stderr
        for line in process.stderr:
            message = line.strip()
            if message:
                yield f"error: {message}\n"
    except Exception as e:
        yield f"error: Error executing command: {str(e)}\n"
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


async def arun_command(command: str, *, work_dir: str = None, separator: str = " ") -> AsyncGenerator[str, None]:
    if not command:
        raise ValueError("Command cannot be empty")
    try:
        if work_dir:
            work_dir = os.path.expanduser(work_dir)
            work_dir = os.path.abspath(work_dir)
        cmd_parts = command.split(separator)
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # Read stdout asynchronously
        while process.returncode is None:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=1.0)
                if not line:
                    break
                message = line.decode('utf-8').strip()
                if message.startswith(("data: ", "result: ", "error: ")):
                    yield f"{message}\n"
                else:
                    yield f"other: {message}\n"
            except asyncio.TimeoutError:
                continue
        # Read any remaining stdout
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            message = line.decode('utf-8').strip()
            if message.startswith(("data: ", "result: ", "error: ")):
                yield f"{message}\n"
            else:
                yield f"other: {message}\n"
        # Read stderr
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            message = line.decode('utf-8').strip()
            if message:
                yield f"error: {message}\n"
        await process.wait()
    except Exception as e:
        yield f"error: Error executing command: {str(e)}\n"
        raise RuntimeError(f"Error executing command: {e}")


async def arun_commands(commands: List[str], work_dir: str = None) -> AsyncGenerator[str, None]:
    for command in commands:
        yield f"command: {command}\n"
        async for line in arun_command(command, work_dir=work_dir):
            yield line
        yield "\n"
