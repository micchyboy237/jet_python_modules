import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()


# Define a signature (input → output spec)
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")


# Create a module (Predict is the simplest)
generate_answer = dspy.Predict(BasicQA)

# Run it
pred = generate_answer(question="What is the capital of Japan?")
print(pred.answer)  # e.g. "Tokyo"
