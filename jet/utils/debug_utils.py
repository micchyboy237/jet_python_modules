def get_non_function_locals():
    # Filter locals to get only non-callable variables, excluding special names
    return {name: value for name, value in locals().items()
            if not callable(value) and not name.startswith('__')}
