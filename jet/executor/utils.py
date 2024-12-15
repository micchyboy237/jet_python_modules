import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Callable


def run_in_threads(tasks, max_workers=3):
    """
    Runs a list of callables in threads.

    :param tasks: List of callables to run.
    :param max_workers: Maximum number of worker threads.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in futures:
            # Wait for task completion and propagate exceptions if any.
            future.result()


def run_in_processes(tasks, max_workers=3):
    """
    Runs a list of callables in separate processes.

    :param tasks: List of callables to run.
    :param max_workers: Maximum number of worker processes.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in futures:
            # Wait for task completion and propagate exceptions if any.
            future.result()


def run_with_pool(tasks, pool_type="multiprocessing", processes=3):
    """
    Executes tasks using a pool (thread or process).

    :param tasks: List of tuples with the format (callable, *args).
    :param pool_type: Type of pool - 'multiprocessing' or 'threading'.
    :param processes: Number of processes/threads in the pool.
    """
    if pool_type == "multiprocessing":
        # Use global function reference instead of a lambda
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.starmap(cpu_pool_task_wrapper, tasks)
            return results
    elif pool_type == "threading":
        with ThreadPoolExecutor(max_workers=processes) as executor:
            futures = [executor.submit(func, *args) for func, *args in tasks]
            return [future.result() for future in futures]
    else:
        raise ValueError(f"Unsupported pool type: {pool_type}")


def cpu_pool_task_wrapper(func, *args, **kwargs):
    """
    Wrapper for multiprocessing.Pool to execute tasks in parallel.

    :param func: Callable to execute.
    :param args: Positional arguments for the callable.
    :param kwargs: Keyword arguments for the callable.
    """
    return func(*args, **kwargs)
