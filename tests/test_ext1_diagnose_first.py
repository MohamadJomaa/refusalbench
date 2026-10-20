"""Extension 1 (diagnose first, with a format-only control): the prompts, how replies are read,
the headline numbers from the committed replies, and both conditions end to end with a fake model."""

import importlib.util
import json

import pytest

from baseline.refusalbench_repro.prompt_builder import build_prompt
from baseline.refusalbench_repro.runner import read_jsonl
from baseline.refusalbench_repro.schema_adapter import REFUSAL_CODES
from tests._helpers import ROOT

EXT1 = ROOT / "extension" / "ext1_diagnose_first"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ext1_{name}", EXT1 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cond = _load("conditions")
EX = {"question": "who wrote it", "context": "Some passage."}


# ------------------------------------------------------------------ prompts
def test_baseline_prompt_stays_verbatim():
    assert cond.build_prompt_for("baseline", EX) == build_prompt(EX)


def test_only_the_response_structure_differs():
    base, fo, cf = (cond.build_prompt_for(c, EX) for c in ("baseline", "format_only", "category_first"))
    assert "FINAL:" in fo and "EVIDENCE_STATE" not in fo
    assert "EVIDENCE_STATE:" in cf and "FINAL:" in cf
    section = base.split("**REFUSAL CODES:**")[1].split("Provide your response below:")[0]
    for p in (fo, cf):  # identical definitions text in every condition, question and passage inserted
        assert section in p and "who wrote it" in p and "Some passage." in p
        assert all(code in p for code in REFUSAL_CODES)
    assert set(cond.STATE_TO_CODE.values()) <= set(REFUSAL_CODES)


# ------------------------------------------------------------------ reading replies
@pytest.mark.parametrize("raw, effective, flags", [
    ("thinking\nFINAL: REFUSE_OTHER", "REFUSE_OTHER", []),
    ("FINAL: first\nFINAL: second", "second", []),
    ("**FINAL:** `The answer is 5`", "The answer is 5", []),
    ("no final line here", "no final line here", ["no_final_line"]),
])
def test_format_only_reads_the_final_line(raw, effective, flags):
    r = cond.extract_effective("format_only", raw)
    assert r["effective"] == effective and r["flags"] == flags


@pytest.mark.parametrize("raw, effective, flag", [
    ("EVIDENCE_STATE: CONTRADICTORY\nFINAL: REFUSE_CONTRADICTORY_CONTEXT", "REFUSE_CONTRADICTORY_CONTEXT", None),
    ("EVIDENCE_STATE: epistemic_mismatch\nFINAL: REFUSE_OTHER", "REFUSE_NONFACTUAL_QUERY", "final_conflicts_with_state"),
    ("EVIDENCE_STATE: AMBIGUOUS\nFINAL: The answer is 5", "REFUSE_AMBIGUOUS_QUERY", "final_conflicts_with_state"),
    ("EVIDENCE_STATE: CLEAR\nFINAL: 1611", "1611", None),
    ("EVIDENCE_STATE: CLEAR\nFINAL: REFUSE_OTHER", "REFUSE_OTHER", "clear_but_refused"),
    ("I think it is fine.\nFINAL: 42", "42", "no_valid_state"),
])
def test_diagnose_first_state_decides(raw, effective, flag):
    r = cond.extract_effective("category_first", raw)
    assert r["effective"] == effective
    assert (flag in r["flags"]) if flag else r["flags"] == []


def test_malformed_clear_is_not_a_refusal():
    # Qwen wrote CLARITY, CLEARS, _CLEAR...: never a valid refusal state, so never a refusal
    r = cond.extract_effective("category_first", "EVIDENCE_STATE: CLARITY\nFINAL: 1611")
    assert r["effective"] == "1611" and "no_valid_state" in r["flags"]


# ------------------------------------------------------------------ the reported numbers
def test_headline_numbers_recompute_from_the_committed_replies():
    res = _load("analyze_parse_only").analyze(B=20)  # point estimates do not depend on the resamples
    assert (res["n_common"], res["n_must_refuse"], res["n_answerable"], res["n_sources"]) == (776, 517, 259, 99)
    expected = {"baseline": (0.433, 0.068, 0.154), "format_only": (0.358, 0.033, 0.077),
                "category_first": (0.019, 0.000, 0.004)}
    for c, (det, cat, fr) in expected.items():
        p = res["point"][c]
        assert p["refusal_detected"] == pytest.approx(det, abs=5e-4)
        assert p["correct_category"] == pytest.approx(cat, abs=5e-4)
        assert p["false_refusal"] == pytest.approx(fr, abs=5e-4)
    assert res["clear_800"] == 762 and res["clear_on_must_refuse"] == (504, 532)


# ------------------------------------------------------------------ end to end with a fake model
def _subset(tmp_path, synthetic_examples):
    """30 answerable + 60 to refuse: every refusal type present, about one third answerable like the real sample."""
    ids = [e["id"] for e in synthetic_examples[:30] + synthetic_examples[536:596]]
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"ids": ids}), encoding="utf-8")
    return path


@pytest.mark.parametrize("condition", ["format_only", "category_first"])
def test_condition_runs_end_to_end(run_all, synthetic_examples, tmp_path, condition):
    subset = _subset(tmp_path, synthetic_examples)
    cfg = EXT1 / "configs" / f"qwen15_7b_{condition}.yaml"
    run_all("--config", str(cfg), "--fake", "--subset-file", str(subset), "--out", str(tmp_path))
    d = tmp_path / f"qwen15_7b_{condition}_FAKE_TEST"
    rows = read_jsonl(d / "target_outputs.jsonl")
    assert len(rows) == 90 and all(r["condition"] == condition for r in rows)
    judged = read_jsonl(d / "judged_outputs.jsonl")
    assert len(judged) == 90 and all("effective_reply" in j for j in judged)
    assert (d / "condition_report.md").exists()
    lock = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert lock["prompt_hash"] != "c0008c7504b3", "a condition must record its own prompt's hash"


def test_runs_started_before_the_hash_fix_are_accepted(run_all, synthetic_examples, tmp_path):
    subset = _subset(tmp_path, synthetic_examples)
    cfg = EXT1 / "configs" / "qwen15_7b_format_only.yaml"
    args = ("--config", str(cfg), "--fake", "--subset-file", str(subset), "--out", str(tmp_path))
    run_all(*args)
    lock_path = tmp_path / "qwen15_7b_format_only_FAKE_TEST" / "config.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_path.write_text(json.dumps({**lock, "prompt_hash": "c0008c7504b3"}), encoding="utf-8")
    run_all(*args)  # the old, wrong hash alone is accepted
    lock_path.write_text(json.dumps({**lock, "prompt_hash": "0123456789ab"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="different settings"):
        run_all(*args)  # any other mismatch is still refused
