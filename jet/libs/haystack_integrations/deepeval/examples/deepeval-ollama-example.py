from haystack import Pipeline

from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric
from deepeval.models import OllamaModel

from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, "
        "number of participants, and economic impact.",
        "Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports "
        "personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people.",
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language.",
        "Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write "
        "clear, logical code for both small and large-scale software projects.",
    ],
]
RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]

pipeline = Pipeline()
evaluator = DeepEvalEvaluator(
    metric=DeepEvalMetric.FAITHFULNESS,
    metric_params={
        "model": OllamaModel(model="qwen3:4b-q4_K_M", temperature=0.0, verbose=False),  # Use Ollama model as the judge LLM
    },
)
pipeline.add_component("evaluator", evaluator)

# Each metric expects a specific set of parameters as input. Refer to the
# DeepEvalMetric class' documentation for more details.
results = pipeline.run({"evaluator": {"questions": QUESTIONS, "contexts": CONTEXTS, "responses": RESPONSES}})

logger.gray("Results:")
for output in results["evaluator"]["results"]:
    logger.success(output)
