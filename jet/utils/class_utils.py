import inspect
from typing import Type


def class_to_string(cls: Type | object) -> str:
    # If the input is an object, get its class first
    if not isinstance(cls, type):
        cls = cls.__class__

    # Get the source code of the class
    class_source = inspect.getsource(cls)

    # Strip leading and trailing whitespace to match desired formatting
    return class_source.strip()


def get_class_name(cls: Type | object) -> str:
    # If the input is an object, get its class first
    if not isinstance(cls, type):
        cls = cls.__class__

    # Return the class name
    return cls.__name__


# Example usage:
if __name__ == "__main__":
    from pydantic import BaseModel
    from typing import Optional

    class CodeSummary(BaseModel):
        features: list[str]
        use_cases: list[str]
        additional_info: Optional[str] = None

    # Stringify the class itself
    class_stringified_version = class_to_string(CodeSummary)
    print(class_stringified_version)

    # Create an object of the class
    code_summary_obj = CodeSummary(
        features=["Sample feature"],
        use_cases=["Sample use case"],
    )

    # Stringify the class of the object
    obj_stringified_version = class_to_string(code_summary_obj)
    print(obj_stringified_version)

    assert class_stringified_version == obj_stringified_version
