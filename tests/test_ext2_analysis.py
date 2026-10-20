"""Extension 2 analysis and scorer maths: metrics checked by hand, thresholds, folds that keep a source
question together, no leakage in the cross-fitting, and the mechanical verdict on synthetic data with a known answer."""

import contextlib
import os
import sys

import numpy as np

from tests._helpers import ROOT

EXT2 = ROOT / "extension" / "ext2_reading_signals"
sys.path.insert(0, str(EXT2))  # the extension's modules import each other by plain name

import analysis as A  # noqa: E402
import scorer as S  # noqa: E402


@contextlib.contextmanager
def _tmp(tmp_path):
    yield str(tmp_path)


rng = np.random.default_rng(1)  # reset by the two tests below, so each sees the original script's draws
n_src, per = 30, 10
source = np.repeat([f"q{i}" for i in range(n_src)], per)
yy = np.tile(np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], bool), n_src)


def make_data(shift, signal=1.5):
    n = len(yy)
    s1 = np.where(yy, signal, -signal) + rng.normal(0, 1.5, n) + shift
    s2 = np.where(yy, signal * 0.5, -signal * 0.5) + rng.normal(0, 1.5, n)
    base = rng.normal(0, 1, (n, 12))
    feats = {k: base + yy[:, None] * (np.arange(12) == i) * 2.0 for i, k in enumerate(("s1_last", "s1_mid", "s2_last", "s2_mid"))}
    return {"ids": [f"id{i}" for i in range(n)], "y": yy, "source": source,
            "utype": np.tile(np.array(["A", "B"]), n // 2), "intensity": np.array(["HIGH"] * n),
            "s1": s1, "s2": s2, "feats": feats}

CFG = {"folds": 5, "seed": 0, "bootstrap": 150, "criterion": "balanced_accuracy", "c_grid": [0.01, 0.1], "pca_components": 8}


def test_metrics_checked_by_hand():
    """Metrics, checked by hand."""
    assert abs(A.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-12  # the classic worked example
    assert A.auroc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0 and A.auroc([1, 1, 0, 0], [1, 2, 3, 4]) == 0.0
    assert A.auroc([0, 1], [1, 1]) == 0.5  # ties count half
    r = A.rates([1, 1, 0, 0], [1, 0, 1, 0])
    assert r == {"frr": 0.5, "mrr": 0.5, "bal_acc": 0.5, "f1": 0.5}, r
    r = A.rates([1, 1, 1, 0, 0], [1, 1, 0, 0, 0])  # tp=2 fp=0 fn=1 -> f1 = 4/5; frr 0; mrr 1/3
    assert r["frr"] == 0.0 and abs(r["mrr"] - 1 / 3) < 1e-12 and abs(r["f1"] - 0.8) < 1e-12 and abs(r["bal_acc"] - 5 / 6) < 1e-12


def test_threshold_choice():
    """Threshold choice."""
    y = np.array([0, 0, 0, 1, 1, 1], bool)
    s = np.array([-3, -2, -1, 1, 2, 3.0])
    t = A.pick_threshold(y, s)
    assert -1 < t < 1 and A.rates(y, s > t)["bal_acc"] == 1.0
    s_noisy = np.array([-3, -2, 1.5, -0.5, 2, 3.0])  # one answerable scores high, one refusal scores low
    for crit in ("balanced_accuracy", "f1", "frr_cap"):
        t = A.pick_threshold(y, s_noisy, crit, frr_cap=0.34)
        assert A.rates(y, s_noisy > t)["frr"] <= 0.34 + 1e-9 or crit != "frr_cap"
    t = A.pick_threshold(y, s_noisy, "frr_cap", frr_cap=0.0)  # no false refusal allowed: threshold above 1.5
    assert A.rates(y, s_noisy > t)["frr"] == 0.0


def test_folds_keep_a_source_question_together():
    """Folds keep a source question together."""
    srcs = np.array([f"q{i % 17}" for i in range(170)])
    f = A.make_folds(srcs, 5, 0)
    for q in set(srcs):
        assert len(set(f[srcs == q])) == 1, "a source question must live in exactly one fold"
    assert set(f) == {0, 1, 2, 3, 4} and max(np.bincount(f)) - min(np.bincount(f)) <= 3 * 10


def test_scorer_maths():
    """Scorer maths."""
    assert S.common_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == [1, 2] and S.common_prefix([[1], [2]]) == []
    logp, lo = S.s1_from_variant_logprobs([np.log(0.2), np.log(0.3)])  # two exclusive openings, 0.5 in total
    assert abs(np.exp(logp) - 0.5) < 1e-12 and abs(lo) < 1e-9
    assert S.log_odds_from_logprobs(np.log(0.9)) > 2.19 and np.isfinite(S.log_odds_from_logprobs(0.0))
    sc, mass = S.s2_from_logprobs([np.log(0.3)], [np.log(0.3)])
    assert abs(sc) < 1e-12 and abs(mass - 0.6) < 1e-12
    v = S.refusal_variants(lambda text: [ord(c) for c in text])  # toy tokenizer: one token per character
    assert v[""] == [ord(c) for c in "REFUSE_"] and v["**"][:2] == [ord("*"), ord("*")]
    assert S.build_s2_prompt({"question": "who {x}", "context": "c {y}"}).count("who {x}") == 1  # braces in data are safe
    assert S.build_s1_prompt({"question": "Q?", "context": "C."}).strip().endswith("ANSWER:")


def test_shards_are_resumable_and_atomic(tmp_path):
    """Shards: resumable and atomic."""
    with _tmp(tmp_path) as d:
        sd = os.path.join(d, "shards")
        exs = [{"id": f"e{i}", "is_answerable": i % 2 == 0, "source_id": f"s{i % 5}", "uncertainty_type": "t",
                "intensity": "HIGH", "question": "q", "context": "c"} for i in range(50)]
        sc_ = S.FakeScorer({})
        S.run_scoring(exs[:30], sc_, sd, "s1", 8, 4, log=lambda m: None)
        assert S.done_ids(sd, "s1") == {e["id"] for e in exs[:30]}
        S.run_scoring(exs, sc_, sd, "s1", 8, 4, log=lambda m: None)  # resume: only the rest is scored
        loaded = S.load_shards(sd, "s1")
        assert set(loaded) == {e["id"] for e in exs} and len(loaded) == 50
        with open(os.path.join(sd, "s1_99999.npz"), "wb") as fh:
            fh.write(b"garbage")  # a half-written file must not break the resume
        assert len(S.done_ids(sd, "s1")) == 50


def test_no_leakage_in_the_cross_fitting():
    """No leakage in the cross-fitting."""
    global rng
    rng = np.random.default_rng(1)
    data = make_data(0.0)
    _, _, thr, _ = A.crossfit(data, CFG, log=lambda m: None)
    fold = A.make_folds(source, 5, 0)
    flipped = dict(data)
    flipped["y"] = data["y"].copy()
    flipped["y"][fold == 2] = ~flipped["y"][fold == 2]  # corrupt ONLY fold 2's labels
    _, _, thr2, _ = A.crossfit(flipped, CFG, log=lambda m: None)
    assert thr["s1"][2] == thr2["s1"][2], "fold 2's threshold must not depend on fold 2's own labels"
    assert any(thr["s1"][k] != thr2["s1"][k] for k in (0, 1, 3, 4)), "other folds do train on fold 2"
    assert thr["s2"][2] == thr2["s2"][2]


def test_full_analysis_signal_vs_no_signal(tmp_path):
    """Full analysis: signal vs no signal."""
    global rng
    rng = np.random.default_rng(1)
    make_data(0.0)  # the draws the original script used for the leakage check
    with _tmp(tmp_path) as d:
        meta = {"model": "fake", "precision": "none", "s1_mode": "variants", "mean_p_refuse_s1": 0.5,
                "mean_yes_no_mass_s2": 1.0, "run_dir": d, "fake": True}
        # signal, but the default threshold (0) is badly placed: S1 shifted down so nothing is refused by default
        res = A.run_analysis(make_data(-4.0), os.path.join(d, "a"), CFG, meta, novelty_mask=np.arange(len(yy)) % 2 == 0,
                             baseline_pred={f"id{i}": 0 for i in range(100)}, log=lambda m: None)
        v, h1, h2 = A.verdict(res["all"])
        assert v == "SUPPORTED" and h1 and h2, (v, res["all"]["auroc"]["s1"], res["all"]["diffs"])
        assert res["all"]["ops"]["s1_default"]["mrr"]["value"] > 0.9 and res["all"]["ops"]["s1_tuned"]["bal_acc"]["value"] > 0.7
        assert res["all"]["auroc"]["s3"]["value"] > 0.6 and res["novelty800"]["n"] == len(yy) // 2
        for fn in ("RESULT_AUTO.md", "metrics.json", "scores.csv", "tradeoff.png", "s1_distribution.png"):
            assert os.path.getsize(os.path.join(d, "a", fn)) > 0, fn
        report = open(os.path.join(d, "a", "RESULT_AUTO.md"), encoding="utf-8").read()
        assert "SUPPORTED" in report and "FAKE SCORER" in report
        # no signal at all: scores are pure noise
        noise = make_data(0.0, signal=0.0)
        noise["feats"] = {k: rng.normal(0, 1, v.shape) for k, v in noise["feats"].items()}
        res0 = A.run_analysis(noise, os.path.join(d, "b"), CFG, meta, log=lambda m: None)
        v0, h10, h20 = A.verdict(res0["all"])
        assert v0 == "NOT SUPPORTED" and not h10, (v0, res0["all"]["auroc"]["s1"])
        # signal and a well-placed default threshold: the signal check passes, the gate check must not
        res1 = A.run_analysis(make_data(0.0), os.path.join(d, "c"), CFG, meta, log=lambda m: None)
        assert A.verdict(res1["all"])[1] is True
