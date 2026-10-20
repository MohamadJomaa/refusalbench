"""The whole baseline pipeline end to end, with a fake model, a fake judge and synthetic data:
generate -> judge (parse first) -> metrics -> report, then resume, then the settings lock."""

import json

import pytest

from baseline.refusalbench_repro.runner import read_jsonl


def test_fake_run_end_to_end(run_all, tmp_path):
    run_all("--fake", "--dev-subset", "20", "--out", str(tmp_path))
    d = tmp_path / "qwen15_7b_baseline_FAKE_TEST"
    for f in ("target_outputs.jsonl", "judged_outputs.jsonl", "metrics.json", "reproduction_report.md",
              "metrics_by_category.csv", "confusion_matrix.csv", "handcheck_sheet.csv", "config.json"):
        assert (d / f).exists(), f
    ids = [r["example_id"] for r in read_jsonl(d / "target_outputs.jsonl")]
    assert len(ids) == 20 == len(set(ids))
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    assert m["overall"]["n_scored"] == 20 and m["prompt_hash"] == "c0008c7504b3"

    # running again resumes: nothing is generated or judged twice
    run_all("--fake", "--dev-subset", "20", "--out", str(tmp_path))
    assert len(read_jsonl(d / "target_outputs.jsonl")) == 20
    assert len(read_jsonl(d / "judged_outputs.jsonl")) == 20


def test_settings_lock_refuses_to_mix_runs(run_all, tmp_path):
    run_all("--fake", "--dev-subset", "20", "--out", str(tmp_path))
    with pytest.raises(SystemExit, match="different settings"):
        run_all("--fake", "--dev-subset", "10", "--out", str(tmp_path))
