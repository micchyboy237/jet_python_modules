import json
from typing import Iterator, List, Optional, TypedDict, Union
from jet.models.model_types import LLMModelType
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.utils.markdown import extract_json_block_content

MODEL_PATH: LLMModelType = "qwen3-1.7b-4bit"
PROMPT_TEMPLATE = """\
System:
Generate unique entity labels for NER from this text, returning a valid JSON array of unique string labels (e.g., ["person", "award", "date", "competitions", "teams"]). Respond only with a single JSON code block. Below is an example to guide the response format:

Example:
Task:
Generate 5 entity labels
Text:
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
Output:
```json
[
    "person",
    "award",
    "date",
    "competitions",
    "teams"
]
```<|im_end|>

Task:
Generate {count} entity labels
Text:
{text}
Output:
"""


def generate_labels(
    texts: Union[str, List[str]],
    model_path: LLMModelType = MODEL_PATH,
    max_labels: int = 5
) -> Union[List[str], List[List[str]]]:
    """Generates unique entity labels for NER from a single text or list of texts.

    Args:
        texts: A single string or list of strings to generate entity labels for.
        model_path: The model to use for label generation (default: MODEL_PATH).
        max_labels: Maximum number of labels to generate per text (default: 5).

    Returns:
        List[str] if input is a single string, or List[List[str]] if input is a list of strings.

    Raises:
        ValueError: If input text is empty or invalid JSON is received from the model.
    """
    client = MLXModelRegistry.load_model(model_path)
    result: List[List[str]] = []

    # Normalize input to a list
    input_texts = [texts] if isinstance(texts, str) else texts

    if not input_texts:
        raise ValueError("Input texts cannot be empty")

    for text in input_texts:
        if not text.strip():
            raise ValueError("Individual text cannot be empty")

        prompt = PROMPT_TEMPLATE.format(text=text.strip(), count=max_labels)
        logger.debug(f"Generating labels for text: {text[:50]}...")

        try:
            response = client.chat(
                prompt,
                model=model_path,
                verbose=True,
                logit_bias=["```json"],
            )
            json_string = extract_json_block_content(response["content"])
            labels = json.loads(json_string)

            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise ValueError(
                    "Model response must be a JSON array of strings")

            # Ensure unique labels and limit to max_labels
            unique_labels = list(dict.fromkeys(labels))[:max_labels]
            result.append(unique_labels)

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response for text: {text[:50]}...: {str(e)}")
            raise ValueError(f"Invalid JSON response from model: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing text: {text[:50]}...: {str(e)}")
            raise

    # Return a single list for single string input, otherwise list of lists
    return result[0] if isinstance(texts, str) else result


# Usage example
if __name__ == "__main__":
    sample_text = "The Apollo 11 mission, launched on July 16, 1969, was the first manned moon landing, with astronauts Neil Armstrong and Buzz Aldrin setting foot on the lunar surface."
    labels = generate_labels(sample_text, model_path="qwen3-1.7b-4bit")
    print("Generated labels:", labels)
    for label in labels:
        print(f"Processing label: {label}")
