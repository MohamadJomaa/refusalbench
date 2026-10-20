"""The scoring rules: reply parser, judge-reply parser and the paper's metrics on a hand-computed case."""

import pytest

from baseline.refusalbench_repro.judge import parse_judge_response
from baseline.refusalbench_repro.metrics import compute_metrics, normalize_classification
from baseline.refusalbench_repro.parser import parse_reply


@pytest.mark.parametrize("reply, expected", [
    ("REFUSE_AMBIGUOUS_QUERY", ("clean_code", "REFUSE_AMBIGUOUS_QUERY")),
    ("**`REFUSE_NONFACTUAL_QUERY`**", ("clean_code", "REFUSE_NONFACTUAL_QUERY")),
    ("REFUSE_OTHER because REFUSE_OTHER", ("clean_code", "REFUSE_OTHER")),
])
def test_one_clean_code_is_scored_by_code(reply, expected):
    assert parse_reply(reply) == expected


@pytest.mark.parametrize("reply, status", [
    ("REFUSE_AMBIGUOUS_QUERY or REFUSE_OTHER", "multiple_codes"),
    ("REFUSE_MADE_UP", "unknown_code"),
    ("REFUSE_INFO_MISSING_IN", "unknown_code"),  # the truncated code Qwen wrote 29 times
    ("I cannot answer this.", "no_code"),         # a plain-words refusal is NOT assumed to be an answer
    ("   ", "empty"),
    (None, "empty"),
])
def test_everything_else_goes_to_the_judge(reply, status):
    assert parse_reply(reply)[0] == status


def test_judge_reply_parser():
    assert parse_judge_response("CLASSIFICATION: answer_attempt\nQUALITY_SCORE: 5\nEXPLANATION: ok")[:2] == (
        "answer_attempt", 5.0)
    assert parse_judge_response("CLASSIFICATION: REFUSE_OTHER\nQUALITY_SCORE: N/A\nEXPLANATION: x")[:2] == (
        "REFUSE_OTHER", None)


def test_judge_reply_fallbacks_are_visible():
    cls, _, _, status = parse_judge_response("garbage mentioning REFUSE_NONFACTUAL_QUERY")
    assert (cls, status) == ("REFUSE_NONFACTUAL_QUERY", "fallback_code")
    assert parse_judge_response("garbage")[3] == "fallback_default"


@pytest.mark.parametrize("raw, expected", [
    ("REFUSE_INFO_MISSING_IN CONTEXT", "REFUSE_INFO_MISSING_IN_CONTEXT"),
    ("REFUSE_INFO_MISSING_INCONTEXT", "REFUSE_INFO_MISSING_IN_CONTEXT"),
    ("REFUSE_SOMETHING_ELSE_ENTIRELY", "REFUSE_OTHER"),
    ("answer_attempt", "answer_attempt"),
    ("EMPTY", "EMPTY"),
    ("REFUSE_AMBIGUOUS_QUERY", "REFUSE_AMBIGUOUS_QUERY"),
])
def test_mistyped_judge_codes_are_normalized(raw, expected):
    assert normalize_classification(raw) == expected


def _row(i, expected, pred, score=None):
    return dict(id=i, expected=expected, answerable=expected == "ANSWER_CORRECTLY", type="t",
                intensity="LOW" if expected == "ANSWER_CORRECTLY" else "HIGH", pred=pred, score=score)


def test_metrics_match_a_hand_computed_case():
    A = "ANSWER_CORRECTLY"
    rows = [
        # 4 answerable: 2 correct answers (score 5 and 4), 1 low-score answer (3), 1 false refusal
        _row(1, A, "answer_attempt", 5.0), _row(2, A, "answer_attempt", 4.0),
        _row(3, A, "answer_attempt", 3.0), _row(4, A, "REFUSE_OTHER"),
        # 4 must refuse: 2 right reason, 1 wrong reason, 1 missed refusal
        _row(5, "REFUSE_AMBIGUOUS_QUERY", "REFUSE_AMBIGUOUS_QUERY"),
        _row(6, "REFUSE_NONFACTUAL_QUERY", "REFUSE_NONFACTUAL_QUERY"),
        _row(7, "REFUSE_NONFACTUAL_QUERY", "REFUSE_OTHER"),
        _row(8, "REFUSE_NONFACTUAL_QUERY", "answer_attempt"),
    ]
    m = compute_metrics(rows)
    assert m["answer_accuracy"] == 2 / 4
    assert m["false_refusal_rate"] == 1 / 4
    assert m["refusal_accuracy"] == 2 / 4
    assert m["missed_refusal_rate"] == 1 / 4
    assert m["overall_refusal_rate_on_unanswerable"] == 3 / 4
    assert m["calibrated_refusal_score"] == 0.5 * 0.5 + 0.5 * 0.5
    # detection, refuse = positive: tp 3, fp 1, fn 1
    assert m["refusal_detection_precision"] == pytest.approx(3 / 4)
    assert m["refusal_detection_recall"] == pytest.approx(3 / 4)
    assert m["refusal_detection_f1"] == pytest.approx(0.75)
