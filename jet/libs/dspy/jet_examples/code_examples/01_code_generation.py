import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()


class CodeGeneration(dspy.Signature):
    """Generate python code to answer the question."""

    question: str = dspy.InputField(description="The question to answer")
    code: dspy.Code["java"] = dspy.OutputField(description="The code to execute")


predict = dspy.Predict(CodeGeneration)

result = predict(question="Given an array, find if any of the two numbers sum up to 10")
print(result.code)
