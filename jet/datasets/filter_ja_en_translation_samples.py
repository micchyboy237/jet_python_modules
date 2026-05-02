import re
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_translation_related(conversations):
    """
    Detect if a conversation is related to Japanese-English translation.
    Returns True if it should be filtered out (i.e., is translation-related).
    """
    full_text = " ".join(msg["value"] for msg in conversations).lower()

    # Common translation indicators
    translation_patterns = [
        r"translate (the following|this text|to english|to japanese|from japanese|from english)",
        r"翻訳してください",  # Japanese for "please translate"
        r"英語に翻訳",  # Translate to English
        r"日本語に翻訳",  # Translate to Japanese
        r"japanese to english",
        r"english to japanese",
        r"bidirectional translation",
        r"parallel text",
        r"translation pair",
        r"以下を英語に訳して",  # Common Japanese prompt patterns
        r"以下を日本語に訳して",
    ]

    for pattern in translation_patterns:
        if re.search(pattern, full_text):
            return True

    # Check for paired structures: English prompt + Japanese equivalent or vice versa
    # Many entries in this dataset are parallel (English + -ja version)
    if len(conversations) >= 1:
        first_msg = conversations[0]["value"]
        # If it looks like a direct translation request or has heavy code-switching
        if any(kw in first_msg.lower() for kw in ["translate", "翻訳", "訳して"]):
            return True

    # Heuristic: very similar structure in bilingual pairs often indicates translation data
    # But this is conservative - adjust based on your needs
    return False


# Load the dataset
dataset = load_dataset("shisa-ai/shisa-v2.1-sharegpt", split="train")

print(f"Original dataset size: {len(dataset)}")

# Filter: keep only samples that are NOT translation-related
filtered_dataset = dataset.filter(
    lambda x: not is_translation_related(x["conversations"]),
    num_proc=4,  # Use multiple processes for speed
)

print(f"Filtered dataset size: {len(filtered_dataset)}")
print(f"Removed {len(dataset) - len(filtered_dataset)} translation-related samples")

# Optional: save the filtered dataset using OUTPUT_DIR
filtered_dataset.save_to_disk(str(OUTPUT_DIR))
print(f"✅ Successfully saved filtered dataset to: {OUTPUT_DIR.resolve()}")

# Or push to Hugging Face (uncomment and fill in your details)
# filtered_dataset.push_to_hub("yourusername/shisa-v2.1-sharegpt-non-translation")
