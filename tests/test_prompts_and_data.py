"""The parts copied from the paper must stay unchanged, and the dataset mapping must not invent anything."""

import pytest

from baseline.refusalbench_repro.judge import build_judge_prompt
from baseline.refusalbench_repro.prompt_builder import PROMPT_TEMPLATE, build_prompt
from baseline.refusalbench_repro.runner import prompt_hash
from baseline.refusalbench_repro.schema_adapter import REFUSAL_CODES, adapt_row
from tests._helpers import synthetic_rows


def test_paper_prompt_is_unchanged():
    # c0008c7504b3 is the hash recorded in every baseline run folder. If this fails, the paper's prompt
    # was edited: saved runs would refuse to resume and the reproduction would no longer be faithful.
    assert prompt_hash() == "c0008c7504b3"
    assert prompt_hash("baseline") == "c0008c7504b3"


def test_prompt_lists_every_refusal_code():
    for code in REFUSAL_CODES:
        assert code in PROMPT_TEMPLATE


def test_prompt_inserts_question_and_passage():
    p = build_prompt({"question": "Who wrote it?", "context": "It was written by Ada."})
    assert "QUERY:\n        Who wrote it?" in p
    assert "CONTEXT PASSAGES:\n        It was written by Ada." in p
    assert p.rstrip().endswith("ANSWER:")


def test_judge_prompt_carries_the_reference_answers():
    p = build_judge_prompt("When?", "In 1611.", ["1611", "the year 1611"])
    assert "MODEL RESPONSE:\nIn 1611." in p and "1611\nthe year 1611" in p
    assert "No reference answers provided" in build_judge_prompt("When?", "REFUSE_OTHER", None)


def test_schema_adapter_maps_released_fields():
    raw = synthetic_rows(n=600, n_answerable=1)
    ans, ref = adapt_row(raw[0]), adapt_row(raw[599])
    assert ans["is_answerable"] and ans["reference_answers"] == ["1611"]
    assert ans["question"] == raw[0]["perturbed_query"] and ans["context"] == raw[0]["perturbed_context"]
    assert ans["source_id"] == raw[0]["source_id"]
    assert not ref["is_answerable"] and ref["reference_answers"] is None
    assert ref["expected_behavior"] in REFUSAL_CODES


def test_schema_adapter_rejects_unknown_labels():
    bad = dict(synthetic_rows(n=1)[0], expected_rag_behavior="REFUSE_INVENTED")
    with pytest.raises(ValueError):
        adapt_row(bad)
