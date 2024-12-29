import inspect
from typing import Type


def class_to_string(cls: Type) -> str:
    # Get the source code of the class
    class_source = inspect.getsource(cls)

    # Strip leading and trailing whitespace to match desired formatting
    return class_source.strip()


# Example usage:
if __name__ == "__main__":
    from pydantic import BaseModel
    from typing import Optional

    class CodeSummary(BaseModel):
        features: list[str]
        use_cases: list[str]
        additional_info: Optional[str] = None

    stringified_version = class_to_string(CodeSummary)
    print(stringified_version)
