from txtai.pipeline import Labels
from tqdm import tqdm
from instruction_generator.helpers.dataset import load_data, save_data, generate_hash
from instruction_generator.utils.time_utils import time_it
from instruction_generator.translation.translator import Translator
from enum import Enum
from instruction_generator.utils.logger import logger
import json

# Create labels model
print("Loading labels model...")
labels = Labels("facebook/bart-large-mnli")

data = load_data(
    "server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/transcription_pairs.json")
output_transcriptions_file = None


@time_it
def convert_tagalog_to_english(text):
    model_name = "Helsinki-NLP/opus-mt-tl-en"

    try:
        translator = Translator(model_name)

        generation_config = {
            "early_stopping": False,
            "num_beams": 4,
            "num_return_sequences": 1,
            "output_scores": False,
        }
        translation_result = translator.generate(text, generation_config)

        return translation_result
    except Exception as e:
        print(f"An error occurred: {e}")


# Get unique labels from data
tags = list(set([item.get('label') for item in data]))
tags = [tag for tag in tags if tag]  # remove None values

if not tags:
    tags = ["Continuation/Elaboration", "Contrast/Disagreement",
            "Cause-Effect", "Topic Shift/New Topic"]
    # tags = ["Entailment", "Neutral", "Contradiction"]


print("\n------------------------------------------------------------------------\n")
print("Sentence Pair Relationship\n")

passed_count = 0
results = []

print("{:<35} {:<25} {:<15}".format("Text", "Label", "Passed"))

pbar = tqdm(data)

batch_size = 20
batch_counter = 0

for idx, item in enumerate(pbar):
    sentence1 = item['sentence1']
    sentence2 = item['sentence2']
    translation_text = None
    translation_text2 = None

    expected = item.get('label', None)

    prev_result = None

    try:
        id = generate_hash(f"{sentence1} {sentence2}")
        label = labels(f"{sentence1}\n{sentence2}", tags)[0][0]
        result = tags[label]

        if not expected:
            results.append({
                "id": id,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": result,
            })

            batch_counter += 1
            if batch_counter >= batch_size:
                if output_transcriptions_file:
                    save_data(output_transcriptions_file, results)
                batch_counter = 0
            continue

        prev_result = result
        passed = result == expected

    except IndexError:
        result = "error"
        passed = False
    finally:
        if expected:
            results.append({
                "id": id,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": expected,
                "result": result,
            })

            batch_counter += 1
            if batch_counter >= batch_size:
                if output_transcriptions_file:
                    save_data(output_transcriptions_file, results)
                batch_counter = 0

            passed_count += 1 if passed else 0

            logger.info(
                f"\n\n---------- Item {idx + 1} ---------- Passed Percent: {passed_count / (idx + 1) * 100:.2f}%; {passed_count}/{idx + 1}")
            print("{:<35} {:<25} {:<15}".format(
                sentence1[:30], "Actual: " + result, ""))
            print("{:<35} {:<25} {:<15}".format(
                sentence2[:30], "Expected: " + expected, ""))
            logger.success(
                f"Passed: {passed}") if passed else logger.error("Failed")
            print("\n")

# Save results to file
if output_transcriptions_file:
    save_data(output_transcriptions_file, results)

if passed_count:
    passed_percent = passed_count / len(data) * 100
    passed_percent_text = f"Passed: {passed_percent:.2f}%; {passed_count}/{len(data)}"
    num_errors = len(
        [result for result in results if result['result'] == "error"])
    num_errors_text = f"Errors: {num_errors}/{len(data)}"
    print("\n")
    logger.success(passed_percent_text)
    logger.error(num_errors_text)
