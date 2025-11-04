import pytest
from typing import Generator
from stanza.server import StartServer
from stanza.server.client import TimeoutException
from jet.libs.stanza.server.stanza_extractor import StanzaExtractor, SentenceInfo, TokenDict
from stanza.models.constituency import tree_reader

TEXT = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

EN_GOLD_TEXT = EN_GOLD = """
Sentence #1 (12 tokens):
Chris wrote a simple sentence that he parsed with Stanford CoreNLP.

Tokens:
[Text=Chris CharacterOffsetBegin=0 CharacterOffsetEnd=5 PartOfSpeech=NNP]
[Text=wrote CharacterOffsetBegin=6 CharacterOffsetEnd=11 PartOfSpeech=VBD]
[Text=a CharacterOffsetBegin=12 CharacterOffsetEnd=13 PartOfSpeech=DT]
[Text=simple CharacterOffsetBegin=14 CharacterOffsetEnd=20 PartOfSpeech=JJ]
[Text=sentence CharacterOffsetBegin=21 CharacterOffsetEnd=29 PartOfSpeech=NN]
[Text=that CharacterOffsetBegin=30 CharacterOffsetEnd=34 PartOfSpeech=WDT]
[Text=he CharacterOffsetBegin=35 CharacterOffsetEnd=37 PartOfSpeech=PRP]
[Text=parsed CharacterOffsetBegin=38 CharacterOffsetEnd=44 PartOfSpeech=VBD]
[Text=with CharacterOffsetBegin=45 CharacterOffsetEnd=49 PartOfSpeech=IN]
[Text=Stanford CharacterOffsetBegin=50 CharacterOffsetEnd=58 PartOfSpeech=NNP]
[Text=CoreNLP CharacterOffsetBegin=59 CharacterOffsetEnd=66 PartOfSpeech=NNP]
[Text=. CharacterOffsetBegin=66 CharacterOffsetEnd=67 PartOfSpeech=.]
""".strip()

@pytest.fixture(scope="module")
def extractor() -> Generator[StanzaExtractor, None, None]:
    """Yield a StanzaExtractor that starts its own CoreNLP server.

    We pass all required arguments to the constructor so that the extractor
    can build and manage the client internally.  This avoids the
    ``client=…`` kwarg that StanzaExtractor does not accept.
    """
    with StanzaExtractor(
        start_server=StartServer.TRY_START,
        endpoint="http://localhost:9000",
        timeout=150_000,          # generous timeout for model loading
        memory="6g",
        annotators=[
            "tokenize", "ssplit", "pos", "lemma", "ner",
            "depparse", "constituency"
        ],
        output_format="json",
        be_quiet=False,
    ) as ext:
        yield ext

class TestStanzaExtractor:
    def test_extract_sentence_info(self, extractor: StanzaExtractor) -> None:
        expected_tokens_len = 12
        result: list[SentenceInfo] = extractor.extract_sentence_info(TEXT)
        assert len(result) == 1
        sent = result[0]
        assert sent["text"] == TEXT.rstrip()
        assert len(sent["tokens"]) == expected_tokens_len
        assert len(sent["dependencies"]) > 0
        assert sent["constituency"] is not None
        first_token: TokenDict = sent["tokens"][0]
        assert first_token["word"] == "Chris"
        assert first_token["pos"] == "NNP"
        assert first_token["characterOffsetBegin"] == 0

    def test_extract_tokensregex(self, extractor: StanzaExtractor) -> None:
        pattern = "([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/"
        result = extractor.extract_tokensregex(TEXT, pattern)
        expected = {
            "sentences": [
                {
                    "0": {
                        "text": "Chris wrote a simple sentence",
                        "begin": 0,
                        "end": 5,
                        "1": {"text": "Chris", "begin": 0, "end": 1},
                    },
                    "length": 1,
                }
            ]
        }
        assert result == expected

    def test_extract_semgrex(self, extractor: StanzaExtractor) -> None:
        pattern = "{word:wrote} >nsubj {}=subject >obj {}=object"
        result = extractor.extract_semgrex(TEXT, pattern, to_words=True)
        expected = [
            {
                "text": "wrote",
                "begin": 1,
                "end": 2,
                "$subject": {"text": "Chris", "begin": 0, "end": 1},
                "$object": {"text": "sentence", "begin": 4, "end": 5},
                "sentence": 0,
            }
        ]
        assert result == expected

    def test_extract_tregex(self, extractor: StanzaExtractor) -> None:
        pattern = "PP < NP"
        result = extractor.extract_tregex(TEXT, pattern)
        expected = {
            "sentences": [
                {
                    "0": {
                        "sentIndex": 0,
                        "characterOffsetBegin": 45,
                        "codepointOffsetBegin": 45,
                        "characterOffsetEnd": 66,
                        "codepointOffsetEnd": 66,
                        "match": "(PP (IN with)\n  (NP (NNP Stanford) (NNP CoreNLP)))\n",
                        "spanString": "with Stanford CoreNLP",
                        "namedNodes": [],
                    }
                }
            ]
        }
        assert result == expected

    def test_extract_tregex_trees(self, extractor: StanzaExtractor) -> None:
        tree_str = "(ROOT (S (NP (NNP Jennifer)) (VP (VBZ has) (NP (JJ blue) (NN skin)))))   (ROOT (S (NP (PRP I)) (VP (VBP like) (NP (PRP$ her) (NNS antennae)))))"
        trees = tree_reader.read_trees(tree_str)
        pattern = "VP < NP"
        result = extractor.extract_tregex_trees(pattern, trees)
        expected = {
            "sentences": [
                {
                    "0": {
                        "sentIndex": 0,
                        "match": "(VP (VBZ has)\n  (NP (JJ blue) (NN skin)))\n",
                        "spanString": "has blue skin",
                        "namedNodes": [],
                    }
                },
                {
                    "0": {
                        "sentIndex": 1,
                        "match": "(VP (VBP like)\n  (NP (PRP$ her) (NNS antennae)))\n",
                        "spanString": "like her antennae",
                        "namedNodes": [],
                    }
                },
            ]
        }
        assert result == expected

    def test_dont_start_no_server(self) -> None:
        """DONT_START with no external server → client stays uninitialized."""
        ext = StanzaExtractor(start_server=StartServer.DONT_START)
        with pytest.raises(RuntimeError, match="Client not initialized"):
            ext._ensure_client()

    def test_timeout(self) -> None:
        """Force a 1 ms timeout → server cannot respond → TimeoutException."""
        with StanzaExtractor(
            start_server=StartServer.TRY_START,
            timeout=1,                     # 1 ms
            annotators="tokenize,ssplit",
        ) as ext:
            with pytest.raises(TimeoutException):
                ext._ensure_client().annotate(TEXT)