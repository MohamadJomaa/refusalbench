"""The baseline numbers we report can be recomputed from the committed run data.
Answers the question: how did you get this number?"""

import io
import json
import zipfile

import pytest

from baseline.refusalbench_repro.metrics import compute_metrics, join_rows
from tests._helpers import ROOT

ZIP = ROOT / "baseline" / "results" / "qwen15_7b_baseline_judged_day1.zip"


def _jsonl(z, name):
    with z.open(f"qwen15_7b_baseline/{name}") as f:
        return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8") if line.strip()]


def test_reported_baseline_numbers_recompute_from_the_data():
    with zipfile.ZipFile(ZIP) as z:
        target, judged = _jsonl(z, "target_outputs.jsonl"), _jsonl(z, "judged_outputs.jsonl")
    assert len(target) == 1560 and len({r["example_id"] for r in target}) == 1560
    rows, unjudged = join_rows(target, judged)
    m = compute_metrics(rows)
    assert (m["n_scored"], m["n_answerable"], m["n_unanswerable"], unjudged) == (770, 261, 509, 790)
    assert m["answer_accuracy"] == pytest.approx(0.690, abs=5e-4)
    assert m["refusal_accuracy"] == pytest.approx(0.114, abs=5e-4)
    assert m["false_refusal_rate"] == pytest.approx(0.165, abs=5e-4)
    assert m["missed_refusal_rate"] == pytest.approx(0.456, abs=5e-4)
    assert m["refusal_detection_f1"] == pytest.approx(0.668, abs=5e-4)
    # every number in the committed metrics.json must equal a recomputation, not only the headline ones
    committed = json.loads((ROOT / "baseline" / "results" / "metrics.json").read_text(encoding="utf-8"))["overall"]
    for key, value in m.items():
        assert committed[key] == pytest.approx(value), key
