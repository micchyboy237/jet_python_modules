from tqdm.notebook import tqdm
import numpy as np
import urllib.request
import yt_dlp

from jet.data.utils import generate_unique_id
from jet.utils.string_utils import capitalize_first_letter, lower_first_letter
from jet.wordnet.words import compare_words


def transcribe_and_translate(model, dataset, options=None):
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    references = []
    transcriptions = []
    translations = []

    for audio, text in tqdm(dataset):
        transcription = model.transcribe(audio, options)["text"]
        translation = model.translate(audio, options)["text"]

        transcriptions.append(transcription)
        translations.append(translation)
        references.append(text)

    return {
        "references": references,
        "transcriptions": transcriptions,
        "translations": translations
    }


def deduplicate_words(words):
    texts = split_texts_from_words(words)

    # Deduplicate texts
    texts = list(set(texts))

    # Find words based on the deduplicated texts
    words = words.copy()
    deduplicated = []
    for text in texts:
        # Iterate words and pop the first item if it matches each word in the text
        for word in text.split():
            for i, w in enumerate(words):
                if word == w['word'].strip():
                    new_word_item = words[i]
                    new_word_text = words[i]['word'].strip()
                    # Check if latest word in deduplicated is the same as the current word, replace it
                    if deduplicated and compare_words(deduplicated[-1]['word'].strip(), new_word_text):
                        deduplicated[-1] = new_word_item
                        # Check if deduplicated[-2] exists
                        if len(deduplicated) > 1:
                            deduplicated[-2]['end'] = new_word_item['start']

                    else:
                        deduplicated.append(new_word_item)

    return deduplicated


def split_texts_from_words(words):
    texts = []

    # Iterate words and add to texts spllited based of current item "end" and next item "start" are not the same
    current_text = ""
    for i, word in enumerate(words):
        prev_ends_in_comma = current_text[-1] == "," if current_text else False
        current_text += word['word'] if not prev_ends_in_comma else lower_first_letter(
            word['word'])
        not_last_word = i < len(words) - 1
        not_continues_sentence = not_last_word and word['end'] != words[i + 1]['start']
        not_ends_in_comma = current_text[-1] != ","
        next_word_letter = words[i + 1]['word'][0] if not_last_word else ""
        next_word_is_not_first_letter_uppercase = next_word_letter.lower() == next_word_letter

        if not_continues_sentence and not_ends_in_comma and next_word_is_not_first_letter_uppercase:
            sentence = current_text.strip()
            # Check if sentence ends with a punctuation
            if sentence[-1] in ('.', '!', '?'):
                texts.append(capitalize_first_letter(sentence))
            else:
                texts.append(sentence)
            current_text = ""
    texts.append(current_text.strip())

    return texts


def extract_texts_words(words):
    texts = []

    # Iterate words and add to texts spllited based of current item "end" and next item "start" are not the same
    current_text = ""
    current_words = []
    for i, word in enumerate(words):
        current_text += word['word']
        current_words.append(word)
        if i < len(words) - 1 and word['end'] != words[i + 1]['start']:
            sentence = current_text.strip()
            # Check if sentence ends with a punctuation
            if sentence[-1] in ('.', '!', '?'):
                current_words[0]['word'] = capitalize_first_letter(
                    current_words[0]['word'])
                texts.append({
                    "text": capitalize_first_letter(sentence),
                    "words": current_words
                })
            current_text = ""
            current_words = []

    # Add last sentence
    if current_words:
        sentence = current_text.strip()
        current_words[0]['word'] = capitalize_first_letter(
            current_words[0]['word'])
        texts.append({
            "text": capitalize_first_letter(sentence),
            "words": current_words
        })

    return texts


def extract_texts_from_transcriptions(transcriptions):
    texts = []
    current_seek = 0
    current_words = []

    for transcription in transcriptions:
        seek = transcription['seek']
        words = transcription['words']

        if seek != current_seek:
            texts_from_words = split_texts_from_words(current_words)
            texts.append({
                'id': generate_unique_id(),
                'seek': current_seek,
                'texts': texts_from_words
            })

            current_words = []
            current_seek = seek
        else:
            current_words.extend(words)
    return texts
