from instruction_generator.helpers.dataset import (
    load_data, load_data_from_directories, load_unique_samples, save_data)
from instruction_generator.utils.logger import logger
from instruction_generator.utils.string import starts_with_whitespace, split_tokenize_text
from instruction_generator.train.inference.CheckpointGenerator import add_vocab_tokens
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
import re
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize_text(text: str):
    # Include apostrophe and the chracters connected after it
    words = split_tokenize_text(text)
    text_starts_with_whitespace = starts_with_whitespace(text)

    for i, word in enumerate(words):
        word_startswith_apostrophe = word.startswith("'")
        if (i == 0 and not text_starts_with_whitespace) \
                or not word.startswith('Ġ') or word[0].islower():
            if not word_startswith_apostrophe:
                updated_word = 'Ġ' + word
                words[i] = updated_word

    return words


def analyze_tokenization(tokenizer, texts):
    unique_words = set()
    unique_subwords = Counter()
    unique_oov_words = set()

    total_unique_words = 0
    total_unique_oov_subwords = 0

    pbar = tqdm(texts, desc="Analyzing tokenization")
    for idx, text in enumerate(texts):
        # Assuming this function extracts words from the text
        words = tokenize_text(text)
        for word in words:
            if word in unique_words or word in unique_oov_words:
                continue  # Skip words already encountered

            if word in tokenizer.get_vocab():
                unique_words.add(word)
                total_unique_words += 1
            else:
                unique_oov_words.add(word)
                tokenized_word = tokenizer.tokenize(word)
                for subword in tokenized_word:
                    unique_subwords[subword] += 1

        batch_size = 100
        if (idx + 1) % batch_size == 0:
            pbar.update(batch_size)
            logger.success(f"Unique Words: {total_unique_words}")
            logger.warning(f"Unique OOV Words: {len(unique_oov_words)}")

    # Calculate total unique subwords
    total_unique_oov_subwords = sum(unique_subwords.values())

    # Detailed subword frequency can also be logged if needed
    for subword, freq in unique_subwords.most_common():
        if freq > 1:
            print(f"Subword: {subword}, Frequency: {freq}")

    # For each text in the array, check if it is in the vocabulary, its length, and its subwords
    print("\n---------")
    for text in texts:
        whole_words = tokenize_text(text)
        oov_words = unique_oov_words.intersection(set(whole_words))

        print(f"Text: \"{text}\"")
        logger.warning(f"OOV Words ({len(oov_words)}): {oov_words}")
        logger.success(f"Matched Words ({len(whole_words)}): {whole_words}")

    # Calculate percentages
    total_tokens = total_unique_words + total_unique_oov_subwords

    # Logging the results
    print("\n---------")
    logger.info(f"Total Tokens: {total_tokens}")
    logger.debug(f"Total Unique Words: {total_unique_words}")
    logger.debug(f"Unique OOV Words: {len(unique_oov_words)}")
    logger.debug(f"Total Unique Subwords: {total_unique_oov_subwords}")

    if total_tokens > 0:
        logger.success(
            f"Percentage of Unique Whole Words: {total_unique_words/total_tokens*100:.2f}%")
        logger.success(
            f"Percentage of Unique OOV Words: {len(unique_oov_words)/total_tokens*100:.2f}%")
        logger.success(
            f"Percentage of Unique OOV Subwords: {total_unique_oov_subwords/total_tokens*100:.2f}%")


def reduce_texts_for_tokens(texts: list[str], new_tokens: list[str], min_freq: int) -> list[str]:
    # Initialize counters and data structures
    token_frequencies = Counter()
    token_to_texts = defaultdict(list)
    selected_texts = []

    # Map tokens to texts in which they appear
    for i, text in enumerate(texts):
        # Assuming simple whitespace tokenization; adjust as necessary
        text_tokens = set(text.split())
        for token in new_tokens:
            if token in text_tokens:
                token_to_texts[token].append(i)

    # Keep track of which tokens still need to be covered
    tokens_needed = set(new_tokens)

    # Greedily select texts until all new tokens are covered with min frequency
    while tokens_needed and token_to_texts:
        # Find the text that covers the most needed tokens
        best_text_idx = max(token_to_texts.keys(), key=lambda token: len(
            set(token_to_texts[token]) & tokens_needed))
        best_text_indices = token_to_texts.pop(best_text_idx)

        # Update frequencies and tokens needed
        for idx in best_text_indices:
            selected_texts.append(texts[idx])
            for token in new_tokens:
                if token in texts[idx]:
                    token_frequencies[token] += 1
                    if token_frequencies[token] >= min_freq:
                        tokens_needed.discard(token)

        # Break if all tokens are covered with minimum frequency
        if not tokens_needed:
            break

    return list(set(selected_texts))  # Remove duplicates


def plot_token_distribution(new_tokens_dict, min_freq=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    filtered_tokens = {token: freq for token,
                       freq in new_tokens_dict.items() if min_freq is None or freq > min_freq}

    # Plot the distribution of new token frequencies
    plt.figure(figsize=(10, 6))
    sns.histplot(list(filtered_tokens.values()), bins=30, kde=False)
    plt.title('Distribution of New Token Frequencies')
    plt.xlabel('Frequency')
    plt.ylabel('Number of New Tokens')
    plt.grid(True)
    plt.show()


def extract_most_freq_new_tokens(samples, tokenizer, min_freq=None):
    texts = [f"{sample['instruction']}\n{sample['input']}\n{sample['output']}"
             for sample in samples]

    # Tokenize the dataset and identify OOV tokens
    token_counter = Counter()
    oov_counter = Counter()
    added_tokens = []

    for text in tqdm(texts, desc="Tokenizing texts"):
        words = tokenize_text(text)
        for word in words:
            token_counter[word] += 1
            if word not in tokenizer.get_vocab():
                oov_counter[word] += 1
                if isinstance(min_freq, int) and oov_counter[word] >= min_freq and word not in added_tokens:
                    added_tokens.append(word)
                    subwords = tokenizer.tokenize(word)
                    logger.success(
                        f"New token \"{word}\"; Subwords ({len(subwords)}): {subwords}")

    # Identify new tokens to add based on frequency and not being in the vocabulary
    # Example threshold
    # new_tokens = [token for token, freq in oov_counter.items()
    #               if freq > min_freq]
    new_tokens_dict = {token: freq for token,
                       freq in oov_counter.items() if min_freq is None or freq > min_freq}

    if not new_tokens_dict:
        logger.error("No new tokens to add.")
        raise ValueError("No new tokens to add.")

    return new_tokens_dict


def main_extract_frequent_new_tokens():
    from typing import List, Dict
    from collections import Counter, defaultdict

    # Example usage
    min_freq = 10  # The minimum frequency for each new token

    # Load the pre-trained model and tokenizer
    # model_name = "server/static/models/dost-asti-gpt2/base_model"
    model_name = "instruction_generator/train/finetune/checkpoints"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    # Example dataset
    samples_path = "server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/_train.json"
    # samples_path = "server/static/models/dost-asti-gpt2/base_model/datasets/train/complete_vocab_instructions_en.json"
    samples = load_data(samples_path)

    new_tokens_dict = extract_most_freq_new_tokens(
        samples, tokenizer, min_freq=min_freq)
    save_path = "server/static/models/dost-asti-gpt2/base_model/datasets/train/new_tokens.json"
    save_data(save_path, new_tokens_dict, write=True)

    # reduced_texts = reduce_texts_for_tokens(texts, new_tokens, min_freq)
    # print(f"Reduced set contains {len(reduced_texts)} texts.")

    # Plot the distribution of new token frequencies
    plot_token_distribution(new_tokens_dict, min_freq)


def main_load_and_setup_new_tokens():
    model_name = "instruction_generator/train/finetune/checkpoints"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    save_path = "server/static/models/dost-asti-gpt2/base_model/datasets/train/new_tokens.json"
    new_tokens_dict = load_data(save_path)

    new_tokens = list(new_tokens_dict.keys())
    new_checkpoint_dir = "instruction_generator/train/finetune/new_tokens_checkpoints"
    if new_tokens:
        add_vocab_tokens(model, tokenizer=tokenizer,
                         checkpoint_dir=new_checkpoint_dir,
                         new_tokens=new_tokens)


def main_analyze_tokenization():
    model_name = "server/static/models/dost-asti-gpt2/base_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples_path = "server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/_train.json"
    # samples_paths = [
    #     "server/static/models/dost-asti-gpt2/base_model/datasets/train/complete_vocab_instructions.json",
    #     "server/static/models/dost-asti-gpt2/base_model/datasets/train/complete_vocab_instructions_en.json",
    # ]
    # samples = load_data_from_directories(samples_paths)
    # samples_path = "server/static/models/dost-asti-gpt2/base_model/datasets/train/complete_vocab_translations.json"
    samples = load_data(samples_path)
    # seed = 42
    # samples = load_unique_samples(samples, seed)
    texts = [f"{sample['instruction']}\n{sample['input']}\n{sample['output']}"
             # texts = [f"{sample['translation.en']}\n{sample['translation.tl']}"
             for sample in samples]
    analyze_tokenization(tokenizer, texts)


if __name__ == "__main__":
    # main_analyze_tokenization()
    # main_extract_frequent_new_tokens()
    main_load_and_setup_new_tokens()
