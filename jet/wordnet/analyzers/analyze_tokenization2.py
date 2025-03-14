import os
import json
from typing import List, Dict
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from tqdm.auto import tqdm
from instruction_generator.helpers.dataset import load_data
from instruction_generator.analyzers.analyze_tokenization import tokenize_text


def tokenize_and_identify_oov(texts: List[Dict], tokenizer: AutoTokenizer) -> Counter:
    """Tokenize texts and identify out-of-vocabulary (OOV) tokens."""
    oov_counter = Counter()
    token_counter = Counter()

    for text in tqdm(texts, desc="Tokenizing and identifying OOV"):
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            token_counter[token] += 1
            if not token.startswith('Ġ'):
                token = 'Ġ' + token
            if token not in tokenizer.get_vocab():
                oov_counter[token] += 1

    return oov_counter


def tokenize_and_filter_incomplete_vocab(samples: List[str], tokenizer: AutoTokenizer, tokenize_fn: callable, threshold: float = 1.0) -> List[str]:
    """Tokenize samples and filter tokens not in the tokenizer's vocabulary."""
    complete_vocab_samples = []
    pbar = tqdm(samples)
    for idx, sample in enumerate(samples):
        text = tokenize_fn(sample)
        tokens = tokenize_text(text)
        # if all(token in tokenizer.get_vocab() for token in tokens):
        # Update the condition if at least 100% of the tokens are in the vocabulary
        if tokens and sum(1 for token in tokens if token in tokenizer.get_vocab()) / len(tokens) >= threshold:
            complete_vocab_samples.append(sample)
            pbar.set_description_str(
                f"Complete vocab samples: {len(complete_vocab_samples)}")
            yield sample
        # Check if idx is every 100th sample
        if (idx + 1) % 100 == 0 or idx + 1 == len(samples):
            pbar.update(100)
    return complete_vocab_samples


def filter_tokens_for_addition(oov_counter: Counter, min_freq: int = 10) -> List[str]:
    """Filter OOV tokens based on frequency and additional criteria."""
    return [token for token, freq in oov_counter.items() if freq >= min_freq]


def add_tokens_to_tokenizer_and_model(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, tokens: List[str], model_dir: str):
    """Add tokens to tokenizer and resize model embeddings."""
    if tokens:
        num_added_tokens = tokenizer.add_tokens(tokens)
        if num_added_tokens > 0:
            tokenizer.save_pretrained(model_dir)
            model.resize_token_embeddings(len(tokenizer))
            print(
                f"Added {num_added_tokens} tokens to the tokenizer and resized model embeddings.")


def main_idenfity_oov_tokens_and_add_to_tokenizer():
    # Configuration
    dataset_path = "server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/_train.json"
    model_name = "server/static/models/dost-asti-gpt2/base_model"
    model_dir = "instruction_generator/train/finetune/checkpoints"
    min_freq = 10  # Minimum frequency for a token to be considered for addition

    # Step 1: Load the dataset
    samples = load_data(dataset_path)

    # Step 2: Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)

    # Step 3: Tokenize dataset and identify OOV tokens
    # context = f"{text['instruction']}\n{text['input']}"
    # completion = text['output']
    texts = [f"{sample['instruction']}\n{sample['input']}\n{sample['output']}"
             for sample in samples]
    oov_counter = tokenize_and_identify_oov(texts, tokenizer)

    # Step 4: Filter tokens for addition
    tokens_for_addition = filter_tokens_for_addition(oov_counter, min_freq)

    # Step 5: Add tokens to tokenizer and model
    add_tokens_to_tokenizer_and_model(
        tokenizer, model, tokens_for_addition, model_dir)

    # Optional: Save the model if you've made significant changes or need to persist the updated state
    model.save_pretrained(model_dir)
    print(f"Model saved to {model_dir} with updated tokenizer.")


if __name__ == "__main__":
    main_idenfity_oov_tokens_and_add_to_tokenizer()
