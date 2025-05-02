from jet.wordnet.sentence import split_sentences
from mlx_lm import load


def merge_texts(text, tokenizer, skip_special_tokens=True, max_length=None):
    # Encode the text into token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)

    # If max_length is None or greater than total tokens, no truncation needed
    if max_length is None or max_length >= total_tokens:
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)
        # Use batch_decode to decode all token IDs at once
        decoded_tokens = tokenizer.batch_decode(
            [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
        )
        # Filter out empty strings
        decoded_tokens = [dt for dt in decoded_tokens if dt]

        return {
            "texts": [text] if text else [],
            "token_counts": [len(token_ids)],
            "tokens": [token_ids],
            "token_strings": [token_strings],
            "decoded_tokens": [decoded_tokens],
            "metadata": {"total_tokens": total_tokens, "is_truncated": False}
        }

    # Get the decoded text to find sentence boundaries
    decoded_text = tokenizer.decode(
        token_ids, skip_special_tokens=skip_special_tokens)

    # Split text into sentences using NLTK
    sentences = split_sentences(decoded_text)

    # Initialize variables for grouping texts
    grouped_texts = []
    grouped_token_ids = []
    selected_token_ids = []
    current_token_count = 0
    current_group = []

    for i, sentence in enumerate(sentences):
        sentence_token_ids = tokenizer.encode(
            sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_token_ids)

        # Check if adding the sentence exceeds max_length
        if current_token_count + sentence_token_count <= max_length:
            selected_token_ids.extend(sentence_token_ids)
            current_token_count += sentence_token_count
            current_group.append(sentence)
        else:
            # If there's a current group, add it to grouped_texts and clear it
            if current_group:
                grouped_texts.append(" ".join(current_group))
                grouped_token_ids.append(selected_token_ids)
                current_group = []  # Clear current_group after adding
                current_token_count = 0  # Reset token count for new group
                selected_token_ids = []  # Reset selected token IDs

            # Try merging with the next sentence if possible
            remaining_tokens = max_length - current_token_count
            if remaining_tokens > 0 and i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                merged_sentence = sentence + " " + next_sentence
                merged_token_ids = tokenizer.encode(
                    merged_sentence, add_special_tokens=False)

                if len(merged_token_ids) <= max_length - current_token_count:
                    selected_token_ids.extend(merged_token_ids)
                    current_token_count += len(merged_token_ids)
                    current_group.append(merged_sentence)
                    # Skip the next sentence since it's merged
                    sentences[i + 1] = ""
                    continue

            # If we can't merge or no space left, start a new group
            if remaining_tokens >= sentence_token_count:
                current_group = [sentence]
                selected_token_ids.extend(sentence_token_ids)
                current_token_count = sentence_token_count
            else:
                break

    # Add the final group if it exists
    if current_group:
        grouped_texts.append(" ".join(current_group))
        grouped_token_ids.append(selected_token_ids)

    grouped_decoded_tokens = []
    grouped_token_strings = []
    token_counts = []
    for token_ids in grouped_token_ids:
        token_counts.append(len(token_ids))
        # Convert selected token IDs to token strings and decoded tokens
        token_strings = tokenizer.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens)
        grouped_token_strings.append(token_strings)
        # Use batch_decode to decode all selected token IDs at once
        decoded_tokens = tokenizer.batch_decode(
            [[tid] for tid in token_ids], skip_special_tokens=skip_special_tokens
        )
        grouped_decoded_tokens.append(decoded_tokens)

    # Prepare metadata
    metadata = {
        "total_tokens": total_tokens,
        "is_truncated": len(grouped_texts) > 1
    }

    return {
        "texts": grouped_texts,
        "token_counts": token_counts,
        "tokens": grouped_token_ids,
        "token_strings": grouped_token_strings,
        "decoded_tokens": grouped_decoded_tokens,
        "metadata": metadata
    }
