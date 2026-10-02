import pytest

from graphhop.extract import (
    ExtractionError,
    LLMExtractor,
    ProperNounExtractor,
    normalise_key,
    split_sentences,
    strip_parenthetical,
)
from graphhop.llm import LLMError

from .conftest import FakeLLM


def names(text):
    ex = ProperNounExtractor().extract(text)
    return [n for s in ex.sentences for n in s.entities]


def test_multiword_names_and_connectors():
    assert names("Tim Cook visited the University of Cambridge in March.") == [
        "Tim Cook",
        "University of Cambridge",
    ]


def test_sentence_initial_stopwords_are_dropped_but_names_kept():
    assert names("In Paris the committee met. The Beatles formed in Liverpool.") == [
        "Paris",
        "Beatles",
        "Liverpool",
    ]
    assert names("However it rained.") == []


def test_months_possessives_and_numbers_are_not_entities():
    assert names("On Monday in January Mary's book sold 3000 copies.") == ["Mary"]


def test_and_does_not_glue_two_entities():
    assert names("Paris and London signed the treaty.") == ["Paris", "London"]


def test_entities_are_deduplicated_per_sentence_by_key():
    assert names("Obama met obama and Obama's staff.") == ["Obama"]


def test_normalise_key_and_parenthetical():
    assert normalise_key("The Beatles") == "beatles"
    assert normalise_key("Mary's") == "mary"
    assert normalise_key("St. John-Paul & Co.") == "st john-paul & co"
    assert strip_parenthetical("Ed Wood (film)") == "Ed Wood"


def test_sentence_split():
    assert split_sentences("One thing. Another thing! A third?") == [
        "One thing.",
        "Another thing!",
        "A third?",
    ]


def test_llm_extractor_parses_and_validates():
    reply = '```json\n{"triples": [{"subject": "Alice Smith", "relation": "founded", "object": "Zorbco"}]}\n```'
    ex = LLMExtractor(FakeLLM(reply)).extract("Alice Smith founded Zorbco in 2001.")
    assert ex.triples == [("Alice Smith", "founded", "Zorbco")]
    assert ex.sentences[0].entities == ["Alice Smith", "Zorbco"]


@pytest.mark.parametrize(
    "reply",
    ["not json", '{"triples": [{"subject": "", "relation": "x", "object": "y"}]}', '{"x": 1}'],
)
def test_llm_extractor_rejects_malformed_output(reply):
    with pytest.raises(ExtractionError):
        LLMExtractor(FakeLLM(reply)).extract("Some text.")


def test_llm_extractor_surfaces_provider_failure():
    class Down:
        def generate(self, *a, **k):
            raise LLMError("down", retryable=True)

    with pytest.raises(ExtractionError):
        LLMExtractor(Down()).extract("Some text.")
