import os
from jet.file.utils import load_file, save_file
from transformers import pipeline
from tqdm import tqdm
from enum import Enum
from jet.logger import logger
from jet.wordnet.pos_tagger import POSTagger

# Create pipeline for zero-shot classification
print("Loading classification pipeline...")
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

data = load_file(
    "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/llm/classification/data/transcriptions.json")
output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
# Convert text to sentence attribute
data = [{"id": item['id'], "sentence": item['text']} for item in data]

# Get unique labels from data
tags = ['Question', 'Continuation', 'Statement']

print("\n------------------------------------------------------------------------\n")
print("Sentence Pair Relationship\n")

passed_count = 0
results = []
tagger = POSTagger()

print("{:<35} {:<25} {:<15}".format("Text", "Label", "Passed"))

pbar = tqdm(data)

batch_size = 20
batch_counter = 0

for idx, item in enumerate(pbar):
    sentence = item['sentence']
    expected = item.get('label')
    has_label = expected in tags
    prev_result = None
    passed = False

    try:
        # Check if sentence ends with a question mark
        if sentence.strip()[-1] == "?":
            result = "Question"
        else:
            pos_results = tagger.process_and_tag(sentence)
            if pos_results[0]['pos'] in ['ADV', 'CCONJ', 'SCONJ']:
                result = "Continuation"
            else:
                result = "Statement"

        passed = result == expected

    except IndexError:
        result = "Error"
        passed = False
    finally:
        obj = {
            "id": item['id'],
            "sentence": sentence,
            "label": result,
        }
        if has_label:
            obj['expected'] = expected
            obj['passed'] = passed
            obj['prev_result'] = prev_result

        results.append(obj)

        batch_counter += 1
        if batch_counter >= batch_size:
            save_file(results, os.path.join(
                output_dir, "sentence_intents.json"))
            batch_counter = 0

    if has_label:
        passed_count += 1 if passed else 0

        logger.info(
            f"\n\n---------- Item {idx + 1} ---------- Passed Percent: {passed_count / (idx + 1) * 100:.2f}%; {passed_count}/{idx + 1}")
        print("{:<35} {:<25} {:<15}".format(
            sentence[:30], "Actual: " + result, ""))
        print("{:<35} {:<25} {:<15}".format(
            "", "Expected: " + expected, "Previous: " + prev_result if prev_result else ""))
        logger.success(
            f"Passed: {passed}") if passed else logger.error("Failed")

        print("\n")

save_file(results, os.path.join(output_dir, "sentence_intents.json"))

num_errors = len([result for result in results if result['label'] == "Error"])
num_errors_text = f"Errors: {num_errors}/{len(data)}"
print("\n")
if passed_count:
    passed_percent = passed_count / len(data) * 100
    passed_percent_text = f"Passed: {passed_percent:.2f}%; {passed_count}/{len(data)}"
    logger.success(passed_percent_text)
if num_errors:
    logger.error(num_errors_text)
