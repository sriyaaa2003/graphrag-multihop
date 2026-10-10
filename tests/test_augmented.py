from graphhop import graphstore
from graphhop.extract import AugmentedExtractor, LLMExtractor, ProperNounExtractor
from graphhop.llm import LLMError

from .conftest import FakeLLM, HashEmbedder

TEXT = "Alice Smith founded Zorbco in 2010. The firm is based in Lund."
GOOD = '{"triples": [{"subject": "Alice Smith", "relation": "founded", "object": "Zorbco"}]}'


def test_augmented_keeps_base_sentences_and_adds_llm_triples():
    ex = AugmentedExtractor(ProperNounExtractor(), LLMExtractor(FakeLLM(GOOD))).extract(TEXT)
    assert ("Alice Smith", "founded", "Zorbco") in ex.triples
    assert any("Alice Smith" in s.entities for s in ex.sentences)


def test_augmented_survives_bad_llm_output_and_counts_it():
    aug = AugmentedExtractor(ProperNounExtractor(), LLMExtractor(FakeLLM("not json")))
    ex = aug.extract(TEXT)
    assert ex.triples == [] and ex.sentences  # base extraction still there
    assert aug.failures == 1 and aug.calls == 1


def test_augmented_survives_llm_outage():
    class Down:
        def generate(self, *a, **k):
            raise LLMError("down", retryable=True)

    aug = AugmentedExtractor(ProperNounExtractor(), LLMExtractor(Down()))
    assert aug.extract(TEXT).sentences and aug.failures == 1


def test_prefetch_runs_each_text_once(db):
    llm = FakeLLM(GOOD)
    aug = AugmentedExtractor(ProperNounExtractor(), LLMExtractor(llm))
    aug.prefetch([TEXT, TEXT, "Bob Jones works at Zorbco."], workers=2)
    assert len(llm.prompts) == 2
    aug.extract(TEXT)  # served from the memo
    assert len(llm.prompts) == 2


def test_typed_edge_is_stored(db):
    aug = AugmentedExtractor(ProperNounExtractor(), LLMExtractor(FakeLLM(GOOD)))
    graphstore.ingest_document(
        db.conn, HashEmbedder(), aug, collection="c", external_id="a", title="Alice Smith", text=TEXT
    )
    rels = {r["relation"] for r in db.conn.execute("SELECT relation FROM edge_contrib")}
    assert "founded" in rels and "cooccurs" in rels
