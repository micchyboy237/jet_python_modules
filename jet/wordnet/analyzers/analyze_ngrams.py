from instruction_generator.ngrams.histogram import TextAnalysis
from tqdm import tqdm
from instruction_generator.helpers.dataset import (
    load_data,
    load_data_from_directories,
    save_data,
    distribute_evenly
)
from instruction_generator.utils.time_utils import time_it
from instruction_generator.evaluators.similarity import filter_different_texts
import os


@time_it
def limit_ngram_occurrences(high_ngram_tl_texts: list[dict[str, float]]) -> list[dict[str, float]]:
    high_ngram_tl_texts = sorted(
        high_ngram_tl_texts, key=lambda x: x['ngram'], reverse=True)
    high_ngram_tl_texts_dict = {item['text']: item for item in high_ngram_tl_texts}

    grouped_texts_by_ngram = {}
    for item in high_ngram_tl_texts:
        ngram = item['ngram']
        if ngram not in grouped_texts_by_ngram:
            grouped_texts_by_ngram[ngram] = []
        grouped_texts_by_ngram[ngram].append(item['text'])

    # Use filter_different_texts for each group texts
    limited_texts = []
    for ngram, texts in grouped_texts_by_ngram.items():
        limited_texts.extend(filter_different_texts(texts))

    limited_texts = [high_ngram_tl_texts_dict[text] for text in limited_texts]

    return limited_texts


def generate_histograms(data):
    output_dir = 'server/static/models/dost-asti-gpt2/base_model/datasets/train/histograms/jet_resume'
    os.makedirs(output_dir, exist_ok=True)
    include_keys = ['instruction', 'input', 'output']
    ta = TextAnalysis(data, include_keys=include_keys)

    most_start_results = ta.generate_histogram(
        is_top=True,
        from_start=True,
        ngram_ranges=[(1, 1), (2, 2)],
        top_n=100,
    )
    least_start_results = ta.generate_histogram(
        is_top=False,
        from_start=True,
        ngram_ranges=[(1, 1), (2, 2)],
        top_n=100,
    )
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(2, 2), (3, 5)],
        top_n=100,
    )
    least_any_results = ta.generate_histogram(
        is_top=False,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(2, 2), (3, 5)],
        top_n=100,
    )

    save_data(os.path.join(output_dir, 'most_common_start.json'),
              most_start_results, write=True)
    save_data(os.path.join(output_dir, 'least_common_start.json'),
              least_start_results, write=True)
    save_data(os.path.join(output_dir, 'most_common_any.json'),
              most_any_results, write=True)
    save_data(os.path.join(output_dir, 'least_common_any.json'),
              least_any_results, write=True)


def analyze_ngrams(texts, texts_dict):
    ta = TextAnalysis(texts)
    most_any_results = ta.generate_histogram(
        is_top=True,
        from_start=False,
        apply_tfidf=True,
        ngram_ranges=[(3, 6)],
        top_n=0.02,
    )
    most_any_results_text_dict = {}
    # Flattem sublists
    results = [item for sublist in most_any_results for item in sublist['tfidf']]
    for result in results:
        most_any_results_text_dict[result['ngram']] = result['score']
    # Find the tl_texts with the longest ngrams
    high_ngram_tl_texts = []
    pbar = tqdm(texts)
    for text in pbar:
        for ngram, score in most_any_results_text_dict.items():
            if ngram in text:
                # Check if each word frequency is above the threshold
                # text_words = get_words(text.lower())
                # passed_freq_threshold = True
                # for word in text_words:
                #     if word in tl_dictionary:
                #         if freq_threshold and tl_dictionary[word] < freq_threshold:
                #             passed_freq_threshold = False
                # if not passed_freq_threshold:
                #     continue

                high_ngram_tl_texts.append({
                    "ngram": ngram,
                    "score": score,
                    "text": text,
                })
                pbar.set_description(
                    f"High-ngram texts: {len(high_ngram_tl_texts)}")
                break

    high_ngram_texts = [item['text'] for item in high_ngram_tl_texts]
    remaining_tl_texts = [texts_dict[text]
                          for text in texts if text not in high_ngram_texts]
    limited_tl_texts = limit_ngram_occurrences(high_ngram_tl_texts)
    filtered_tl_texts = [texts_dict[item['text']] for item in limited_tl_texts]
    all_texts = filtered_tl_texts + remaining_tl_texts

    return all_texts


def main_analyze_ngrams():
    data_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/jet_resume.json'
    output_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/train/jet_resume.json'

    data = load_data(data_file)
    texts_dict = {}
    for item in data:
        text = item['instruction'] + ' ' + item['input'] + ' ' + item['output']
        texts_dict[text] = item
    texts = list(texts_dict.keys())

    filtered_data = analyze_ngrams(texts, texts_dict)
    save_data(output_file, filtered_data, write=True)


def main_generate_histograms():
    data_file = 'server/static/models/dost-asti-gpt2/base_model/datasets/foundational1/jet_resume.json'
    data = load_data(data_file)
    generate_histograms(data)


if __name__ == '__main__':
    # main_analyze_ngrams()
    main_generate_histograms()
