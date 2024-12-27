import inspect


def get_initial_function():
    """
    Traces back the stack to find the initial function that started the run.

    Returns:
        str: The name of the initial function and the module it resides in.
    """
    # Get the current stack trace
    stack = inspect.stack()
    # The first frame is the current function, and the last frame is the initial function
    initial_frame = stack[-1]
    # Extract function name and module
    function_name = initial_frame.function
    module = inspect.getmodule(initial_frame.frame)
    module_name = module.__name__ if module else "<module>"
    return f"{module_name}.{function_name}"


# Example usage:
if __name__ == "__main__":
    def test_function():
        print(get_initial_function())

    def start():
        test_function()

    start()
