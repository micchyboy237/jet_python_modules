import json
from instruction_generator.helpers.dataset import load_data_from_directories
from instruction_generator.helpers.words import get_words
from instruction_generator.wordnet.constants import PREFIX_SET, INFIX_SET, SUFFIX_SET
from tqdm import tqdm

# Load Tagalog words
tl_translation_directories = ["server/static/datasets/translations"]
includes = []
excludes = []
data_samples = load_data_from_directories(
    tl_translation_directories, includes, excludes)

# Sets of prefixes, infixes, and suffixes
prefixes = PREFIX_SET
infixes = INFIX_SET
suffixes = SUFFIX_SET


def count_morphemes_in_sample(sample, morpheme_details, morpheme_list, morpheme_type):
    for word in get_words(sample):
        for morpheme in morpheme_list:
            if ((morpheme_type == "prefix" and word.startswith(morpheme)) or
                (morpheme_type == "infix" and morpheme in word) or
                    (morpheme_type == "suffix" and word.endswith(morpheme))):

                if morpheme not in morpheme_details:
                    morpheme_details[morpheme] = {"count": 0, "words": {}}
                morpheme_details[morpheme]["count"] += 1
                if word in morpheme_details[morpheme]["words"]:
                    morpheme_details[morpheme]["words"][word] += 1
                else:
                    morpheme_details[morpheme]["words"][word] = 1


# Initialize dictionaries to store affix details
prefix_details = {}
infix_details = {}
suffix_details = {}

# Process each translation
pbar = tqdm(data_samples, desc="Processing translations")
for item in pbar:
    sample = item["translation.tl"]
    count_morphemes_in_sample(sample, prefix_details, prefixes, "prefix")
    count_morphemes_in_sample(sample, infix_details, infixes, "infix")
    count_morphemes_in_sample(sample, suffix_details, suffixes, "suffix")

    # Update progress bar postfix with current counts
    current_counts = {
        "prefixes": sum(prefix_details[morpheme]["count"] for morpheme in prefix_details),
        "infixes": sum(infix_details[morpheme]["count"] for morpheme in infix_details),
        "suffixes": sum(suffix_details[morpheme]["count"] for morpheme in suffix_details)
    }
    pbar.set_description_str(str(current_counts))

# Combine results in the desired format
affix_data = {
    "prefix": prefix_details,
    "infix": infix_details,
    "suffix": suffix_details
}

output_file = 'instruction_generator/wordnet/datasets/affix_data_tl_translations.json'

# Save to a JSON file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(affix_data, file, ensure_ascii=False, indent=4)

print(f"Data saved to {output_file}")
