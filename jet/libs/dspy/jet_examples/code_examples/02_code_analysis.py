import inspect

import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()


class CodeAnalysis(dspy.Signature):
    """Analyze the time complexity of the function."""

    code: dspy.Code["python"] = dspy.InputField(description="The function to analyze")
    result: str = dspy.OutputField(description="The time complexity of the function")


predict = dspy.Predict(CodeAnalysis)


def sleepsort(x):
    import time

    for i in x:
        time.sleep(i)
        print(i)


result = predict(code=inspect.getsource(sleepsort))
print(result.result)
