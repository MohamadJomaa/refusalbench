"""Extension 2 comparison with the judged baseline: the BETTER / WORSE / NO CLEAR DIFFERENCE rule on synthetic
cases with a known answer, how the judge's labels are read, partial judging, and the reported numbers recomputed
from the committed data."""

import csv
import json
import os
import sys

import numpy as np
import pytest

from tests._helpers import ROOT

EXT2 = ROOT / "extension" / "ext2_reading_signals"
sys.path.insert(0, str(EXT2))  # the extension's modules import each other by plain name

import analysis as A  # noqa: E402
import compare_to_baseline as C  # noqa: E402

rng = np.random.default_rng(3)
N_SRC, PER = 40, 10
n = N_SRC * PER
source = np.repeat([f"q{i}" for i in range(N_SRC)], PER)
y = np.tile(np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], bool), N_SRC)
utype = np.tile(np.array(["A", "B"]), n // 2)
ids = [f"id{i}" for i in range(n)]
good = np.where(y, 2.0, -2.0) + rng.normal(0, 1.5, n)  # a scorer with real signal
pred_good = good > 0


def write_scores(d, s3_pred):
    with open(os.path.join(d, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "uncertainty_type", "intensity", "should_refuse", "fold", "s1", "s2", "s3_heldout",
                    "s1_default", "s1_tuned", "s2_tuned", "s3_tuned"])
        for i in range(n):
            w.writerow([ids[i], source[i], utype[i], "HIGH", int(y[i]), 0, f"{good[i]:.4f}", f"{good[i]:.4f}",
                        f"{good[i]:.4f}", int(pred_good[i]), int(pred_good[i]), int(pred_good[i]), int(s3_pred[i])])


def write_baseline(d, refuse, judged_ids=None, code="REFUSE_OTHER", empty_for=()):
    with open(os.path.join(d, "target_outputs.jsonl"), "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"example_id": i}) + "\n")
    with open(os.path.join(d, "judged_outputs.jsonl"), "w", encoding="utf-8") as f:
        for k, i in enumerate(ids):
            if judged_ids is not None and i not in judged_ids:
                continue
            cls = "EMPTY" if k in empty_for else (code if refuse[k] else "answer_attempt")
            f.write(json.dumps({"example_id": i, "classification": cls}) + "\n")


def test_verdict_rule_and_label_reading(tmp_path):
    """Coin-flip baseline: rules BETTER. Identical baseline: no difference. Oracle baseline: rules WORSE.
    Any REFUSE_* label (even mistyped) refuses; EMPTY and answer_attempt answer. Partial judging warns when the
    judged part is not representative."""
    tmp = str(tmp_path)
    nov, base = os.path.join(tmp, "nov"), os.path.join(tmp, "base")
    os.makedirs(nov), os.makedirs(base)
    write_scores(nov, pred_good)

    def run(refuse, **kw):
        write_baseline(base, refuse, **kw)
        dec, n_gen, n_judged, n_empty = C.load_baseline(base)
        d = C.build_arrays(C.load_scores(nov), dec, kw.get("subset"))
        return d, C.bootstrap(d, 300, 0), (n_gen, n_judged, n_empty)

    # --- a coin-flip baseline: the good rules must be judged BETTER
    d, res, _ = run(rng.random(n) < 0.5)
    assert d["y"].shape[0] == n and res["n_sources"] == N_SRC
    for k in ("s1_tuned", "s2_tuned", "s3_tuned", "s1_default"):
        assert C.judgement(res["rules"][k]["diff"]["bal_acc"]) == "BETTER", k

    # --- a baseline identical to the rules: no difference at all
    same = pred_good.copy()
    d, res, _ = run(same)
    dd = res["rules"]["s3_tuned"]["diff"]["bal_acc"]
    assert dd["value"] == 0.0 and dd["ci"] == [0.0, 0.0] and C.judgement(dd) == "NO CLEAR DIFFERENCE"

    # --- an oracle baseline: the rules must be judged WORSE
    d, res, _ = run(y.copy())
    assert C.judgement(res["rules"]["s3_tuned"]["diff"]["bal_acc"]) == "WORSE"
    assert res["rules"]["baseline"]["bal_acc"]["value"] == 1.0

    # --- how the judge's labels are read: any REFUSE_* code refuses, EMPTY and answer_attempt answer
    refuse = np.zeros(n, bool)
    refuse[:5] = True
    dec, _, _, n_empty = (lambda: (write_baseline(base, refuse, empty_for={0, 1}), C.load_baseline(base))[1])()
    assert dec[ids[0]] is False and dec[ids[1]] is False and n_empty == 2, "EMPTY must count as answering"
    assert dec[ids[2]] is True and dec[ids[4]] is True and dec[ids[5]] is False
    write_baseline(base, refuse, code="REFUSE_CONTRADICTORY_CONTEXT")
    assert C.load_baseline(base)[0][ids[2]] is True

    # --- a mistyped refusal code still counts as a refusal
    write_baseline(base, np.zeros(n, bool))
    rows_ = [json.loads(l) for l in open(os.path.join(base, "judged_outputs.jsonl"), encoding="utf-8")]
    rows_[0]["classification"] = "REFUSE_INFO_MISSING_IN CONTEXT"
    with open(os.path.join(base, "judged_outputs.jsonl"), "w", encoding="utf-8") as f:
        for r in rows_:
            f.write(json.dumps(r) + chr(10))
    assert C.load_baseline(base)[0][ids[0]] is True and C.load_baseline(base)[0][ids[1]] is False

    # --- only part of the baseline is judged: compare on the overlap, and warn if it is not representative
    part = set(ids[: n // 2])
    d, res, (n_gen, n_judged, _) = run(rng.random(n) < 0.5, judged_ids=part)
    assert res["n"] == n // 2 and n_gen == n and n_judged == n // 2
    rows = C.load_scores(nov)
    assert C.coverage_warning(rows, set(ids)) == ""  # everything judged: nothing to warn about
    biased = {r["id"] for r in rows if r["uncertainty_type"] == "A" and r["should_refuse"] == "1"}
    assert "NOT a representative" in C.coverage_warning(rows, biased)

    # --- subset restriction
    sub = set(ids[:100])
    write_baseline(base, rng.random(n) < 0.5)
    d = C.build_arrays(rows, C.load_baseline(base)[0], sub)
    assert len(d["ids"]) == 100

    # --- report and (when matplotlib exists) the figure
    write_baseline(base, rng.random(n) < 0.5)
    dec, n_gen, n_judged, n_empty = C.load_baseline(base)
    d = C.build_arrays(rows, dec)
    res = C.bootstrap(d, 200, 0)
    text = C.render(res, {"baseline_dir": base, "n_generated": n_gen, "n_judged": n_judged, "n_empty": n_empty}, "")
    assert "BETTER" in text and "Uses labels?" in text and "no labels used" in text and "WARNING" not in text
    assert "WARNING" in C.render(res, {"baseline_dir": base, "n_generated": 1, "n_judged": 1, "n_empty": 0}, "oops")
    try:
        import matplotlib  # noqa: F401

        C.plot(os.path.join(tmp, "p.png"), d, res)
        assert os.path.getsize(os.path.join(tmp, "p.png")) > 0
    except ImportError:
        pass  # matplotlib not installed: the figure is not tested


BASELINE_ZIP = ROOT / "baseline" / "results" / "qwen15_7b_baseline_judged_day1.zip"
RESULTS = EXT2 / "results"


def test_auroc_of_the_three_readouts_recomputes_from_scores_csv():
    """Slide 10, panel A: AUROC 0.726 / 0.751 / 0.797 on all 1,600, from the committed held-out scores."""
    rows = C.load_scores(str(RESULTS))
    assert len(rows) == 1600 and len({r["source"] for r in rows}) == 100
    y = np.array([r["should_refuse"] == "1" for r in rows])
    for col, expected in (("s1", 0.726), ("s2", 0.751), ("s3_heldout", 0.797)):
        assert A.auroc(y, np.array([float(r[col]) for r in rows])) == pytest.approx(expected, abs=5e-4), col


def test_comparison_with_the_judged_baseline_recomputes():
    """Slide 10, panel B: on the 770 graded baseline replies, baseline 0.690 and the rules' differences."""
    dec, n_gen, n_judged, _ = C.load_baseline(str(BASELINE_ZIP))  # read straight from the committed zip
    assert (n_gen, n_judged) == (1560, 770)
    d = C.build_arrays(C.load_scores(str(RESULTS)), dec)
    res = C.bootstrap(d, 50, 0)  # point estimates do not depend on the number of resamples
    assert res["n"] == 770 and res["n_sources"] == 99
    assert res["rules"]["baseline"]["bal_acc"]["value"] == pytest.approx(0.690, abs=5e-4)
    expected = {"s1_default": -0.031, "s1_tuned": -0.018, "s2_default": -0.020, "s2_tuned": -0.021, "s3_tuned": 0.028}
    for rule, diff in expected.items():
        assert res["rules"][rule]["diff"]["bal_acc"]["value"] == pytest.approx(diff, abs=5e-4), rule


def test_committed_comparison_matches_its_intervals():
    """The committed comparison.json (2,000 resamples) holds the intervals quoted on slide 10."""
    res = json.loads((RESULTS / "comparison_all" / "comparison.json").read_text(encoding="utf-8"))["results"]
    s3 = res["rules"]["s3_tuned"]["diff"]["bal_acc"]
    assert s3["ci"] == pytest.approx([-0.014, 0.065], abs=5e-4) and C.judgement(s3) == "NO CLEAR DIFFERENCE"
    assert C.judgement(res["rules"]["s1_default"]["diff"]["bal_acc"]) == "WORSE"


def test_per_type_numbers_in_the_comparison():
    """The per-type numbers quoted in the slide 10 notes: the probe finds many more contradictions and
    ambiguities than the baseline, but refuses more answerable questions."""
    dec, *_ = C.load_baseline(str(BASELINE_ZIP))
    table = "\n".join(C.by_type_table(C.build_arrays(C.load_scores(str(RESULTS)), dec)))
    assert "| P-Contradiction | 90 / 42 | 0.71 / 0.17 | 0.23 / 0.50 | 0.19 / 0.29 |" in table
    assert "| P-Ambiguity | 73 / 47 | 0.74 / 0.09 | 0.55 / 0.21 | 0.45 / 0.23 |" in table
    assert "| P-FalsePremise | 88 / 40 | 0.19 / 0.33 | 0.16 / 0.47 | 0.25 / 0.23 |" in table
