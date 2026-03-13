import dspy
from jet.libs.dspy.custom_config import configure_dspy_lm

configure_dspy_lm()


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")


generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input alongside a hint.
question = "What is the color of the sky?"
pred = generate_answer(question=question)
print(pred.answer)
