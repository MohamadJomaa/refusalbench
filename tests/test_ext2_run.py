"""Extension 2 end to end with a fake scorer and synthetic data: scoring -> shards -> cross-fitted analysis -> report.
And, when torch and transformers are installed, the scorer's mechanics on a tiny random Qwen2 model."""

import importlib.util
import json
import os

import pytest

from tests._helpers import ROOT

EXT2 = ROOT / "extension" / "ext2_reading_signals"


def _load_run_ext2(monkeypatch, examples):
    spec = importlib.util.spec_from_file_location("run_ext2_under_test", EXT2 / "run_ext2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "load_refusalbench_nq", lambda: [dict(e) for e in examples])
    return mod


def test_fake_run_end_to_end_and_resume(monkeypatch, synthetic_examples, tmp_path):
    mod = _load_run_ext2(monkeypatch, synthetic_examples)
    args = ["run_ext2.py", "--fake", "--limit", "60", "--out", str(tmp_path)]
    monkeypatch.setattr("sys.argv", args)
    mod.main()
    d = tmp_path / "novelty2_confidence_threshold_FAKE_TEST"
    for f in ("RESULT_AUTO.md", "metrics.json", "scores.csv", "tradeoff.png", "s1_distribution.png", "config.json"):
        assert (d / f).exists() and os.path.getsize(d / f) > 0, f
    assert "FAKE SCORER" in (d / "RESULT_AUTO.md").read_text(encoding="utf-8")
    n_shards = len(os.listdir(d / "shards"))
    monkeypatch.setattr("sys.argv", args)
    mod.main()  # resume: everything is already scored, nothing new is written
    assert len(os.listdir(d / "shards")) == n_shards


def test_prompts_are_unchanged_since_the_reported_run():
    """The prompts hash recorded by the run we report must equal the code's, so that run can be resumed."""
    import sys
    import zipfile

    sys.path.insert(0, str(EXT2))
    import scorer

    with zipfile.ZipFile(EXT2 / "results" / "ext2_run.zip") as z:
        saved = json.loads(z.read("novelty2_confidence_threshold/config.json"))
    assert scorer.prompts_hash() == saved["prompts_hash"]


def test_scorer_on_a_tiny_random_model():
    """Needs torch, transformers and the Qwen tokenizer (a few MB download). Skipped when they are missing."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import runpy

    try:
        runpy.run_path(str(ROOT / "tests" / "ext2_tiny_model_check.py"), run_name="__main__")
    except OSError as exc:  # no network for the tokenizer files
        pytest.skip(f"Qwen tokenizer not available: {exc}")
