"""
Functions to visualize dependency relations in texts and Stanza documents 
"""

from typing import List
from stanza.models.common.constant import is_right_to_left
import stanza
import spacy
from spacy import displacy
from spacy.tokens import Doc
from tqdm import tqdm


def visualize_doc(doc, language) -> List[str]:
    """
    Takes in a Stanza Document and returns dependency visualization HTML strings per sentence.

    The document to visualize must be from the stanza pipeline.

    Right-to-left languages such as Arabic are displayed right-to-left based on the language code.
    """
    visualization_options = {"compact": True, "bg": "#09a3d5", "color": "white", "distance": 90,
                             "font": "Source Sans Pro", "arrow_spacing": 25}
    # blank model - we don't use any of the model features, just the viz
    nlp = spacy.blank("en")
    sentences_to_visualize = []
    for sentence in doc.sentences:
        words, lemmas, heads, deps, tags = [], [], [], [], []
        if is_right_to_left(language):  # order of words displayed is reversed, dependency arcs remain intact
            sent_len = len(sentence.words)
            for word in reversed(sentence.words):
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:  # spaCy head indexes are formatted differently than that of Stanza
                    heads.append(sent_len - word.id)
                else:
                    heads.append(sent_len - word.head)
        else:   # left to right rendering
            for word in sentence.words:
                words.append(word.text)
                lemmas.append(word.lemma)
                deps.append(word.deprel)
                tags.append(word.upos)
                if word.head == 0:
                    heads.append(word.id - 1)
                else:
                    heads.append(word.head - 1)
        document_result = Doc(nlp.vocab, words=words, lemmas=lemmas, heads=heads, deps=deps, pos=tags)
        sentences_to_visualize.append(document_result)

    html_strings: List[str] = []
    for line in sentences_to_visualize:  # render all sentences through displaCy
        html = displacy.render(line, style="dep", options=visualization_options)
        html_strings.append(html)
    return html_strings


def visualize_str(text, pipeline_code, pipe) -> List[str]:
    """
    Takes a string and returns dependency visualization HTML strings using displacy.

    The string is processed using the Stanza pipeline and its dependencies are formatted
    into a spaCy Doc object for visualization. Accepts valid Stanza (UD) pipelines as the
    pipeline argument. Must supply the language code (e.g., 'en') and the Stanza pipeline object.
    """
    doc = pipe(text)
    return visualize_doc(doc, pipeline_code)


def visualize_docs(docs, lang_code) -> List[List[str]]:
    """
    Takes in a list of Stanza document objects and a language code (ex: 'en' for English) and
    returns dependency visualization HTML strings for each document.

    This function uses spaCy visualizations. See the `visualize_doc` function for details.
    """
    html_strings_matrix: List[List[str]] = []
    for doc in docs:
        html_strings_matrix.append(visualize_doc(doc, lang_code))
    return html_strings_matrix


def visualize_strings(texts, lang_code) -> List[List[str]]:
    """
    Takes a language code (ex: 'en' for English) and a list of strings, returning dependency
    visualization HTML strings for each text.

    This function loads the Stanza pipeline for the given language and uses it to visualize
    all of the strings provided.
    """
    pipe = stanza.Pipeline(lang_code, processors="tokenize,pos,lemma,depparse")
    html_strings_matrix: List[List[str]] = []
    for text in tqdm(texts, desc="Visualizing dependencies", unit="text"):
        html_strings_matrix.append(visualize_str(text, lang_code, pipe))
    return html_strings_matrix


def main():
    ar_strings = ['برلين ترفض حصول شركة اميركية على رخصة تصنيع دبابة "ليوبارد" الالمانية', "هل بإمكاني مساعدتك؟",
               "أراك في مابعد", "لحظة من فضلك"]
    en_strings = ["This is a sentence.",
                  "Barack Obama was born in Hawaii. He was elected President of the United States in 2008."]
    zh_strings = ["中国是一个很有意思的国家。"]
    # Testing with right to left language
    visualize_strings(ar_strings, "ar")
    # Testing with left to right languages
    visualize_strings(en_strings, "en")
    visualize_strings(zh_strings, "zh")

if __name__ == '__main__':
    main()
