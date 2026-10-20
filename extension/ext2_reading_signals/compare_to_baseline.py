"""Fair comparison: the judged baseline vs the Extension 2 decision rules, on the SAME examples.

Run this after the baseline is judged (the baseline run folder needs target_outputs.jsonl and judged_outputs.jsonl).

  python extension/ext2_reading_signals/compare_to_baseline.py --out <folder>
      [--baseline-dir <baseline run folder, or its zip>] [--novelty-dir <folder with scores.csv>]
      [--subset-file data/novelty_sample_ids.json]

By default it reads the committed data: the graded baseline zip in baseline/results/ and the committed
scores.csv in this folder's results/, so the reported comparison can be recomputed without a GPU or the judge.

How the baseline decides: a judged reply counts as a refusal when the judge classified it as any REFUSE_* code
(the paper's definition, REFUSE_OTHER included; a mistyped code such as `REFUSE_INFO_MISSING_IN CONTEXT` is mapped to the closest valid code); "answer_attempt" and "EMPTY" count as answering.
Free-text refusals the judge recognised are therefore counted, which the no-judge reference could not do.

Rule fixed before the numbers are seen: a rule is "BETTER than the baseline" when the lower end of the 95% cluster-bootstrap
interval of (rule - baseline) in balanced accuracy is above 0, "WORSE" when the upper end is below 0, and otherwise
"NO CLEAR DIFFERENCE". Balanced accuracy = 1 - (false refusal rate + missed refusal rate) / 2.

Honest labelling in the report: S1/S2 with their default thresholds use no labels. The tuned thresholds and the S3
probe were fitted on the benchmark's labels (cross-fitted over source questions, so no leakage), and the baseline was not.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = HERE                                   # extension/ext2_reading_signals
ROOT = os.path.dirname(os.path.dirname(EXP))  # the repository root
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from analysis import BLUE, ORANGE, AQUA, INK, MUTED, SURFACE, _style, rates  # noqa: E402
from baseline.refusalbench_repro.metrics import normalize_classification  # noqa: E402
from baseline.refusalbench_repro.runner import read_jsonl  # noqa: E402
from baseline.refusalbench_repro.schema_adapter import REFUSAL_CODES  # noqa: E402

RULES = [("s1_default", "S1 default threshold (no labels used)"), ("s1_tuned", "S1 tuned threshold"),
         ("s2_default", "S2 default (No > Yes, no labels used)"), ("s2_tuned", "S2 tuned threshold"),
         ("s3_tuned", "S3 hidden-state probe")]
METRICS = ("frr", "mrr", "bal_acc", "f1")


# ------------------------------------------------------------------ loading
def load_scores(novelty_dir):
    with open(os.path.join(novelty_dir, "scores.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def _read_run_file(baseline_dir, name):
    """A file of a baseline run, from its folder or straight from the zip we commit (e.g. baseline/results/*.zip)."""
    if str(baseline_dir).endswith(".zip"):
        import io
        import zipfile

        with zipfile.ZipFile(baseline_dir) as z:
            member = next((m for m in z.namelist() if m.endswith("/" + name) or m == name), None)
            if member is None:
                return []
            with z.open(member) as f:
                return [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8") if line.strip()]
    return read_jsonl(os.path.join(baseline_dir, name))


def load_baseline(baseline_dir):
    """Returns ({id: refuses?}, n_generated, n_judged, n_empty)."""
    target = _read_run_file(baseline_dir, "target_outputs.jsonl")
    judged = {r["example_id"]: r for r in _read_run_file(baseline_dir, "judged_outputs.jsonl")}
    dec = {i: (normalize_classification(r["classification"]) in REFUSAL_CODES) for i, r in judged.items()}
    n_empty = sum(1 for r in judged.values() if r["classification"] == "EMPTY")
    return dec, len(target), len(judged), n_empty


def build_arrays(score_rows, base_dec, subset_ids=None):
    keep = [r for r in score_rows if r["id"] in base_dec and (subset_ids is None or r["id"] in subset_ids)]
    y = np.array([r["should_refuse"] == "1" for r in keep])
    s2 = np.array([float(r["s2"]) for r in keep])
    pred = {
        "baseline": np.array([base_dec[r["id"]] for r in keep]),
        "s1_default": np.array([r["s1_default"] == "1" for r in keep]),
        "s1_tuned": np.array([r["s1_tuned"] == "1" for r in keep]),
        "s2_default": s2 > 0.0,
        "s2_tuned": np.array([r["s2_tuned"] == "1" for r in keep]),
        "s3_tuned": np.array([r["s3_tuned"] == "1" for r in keep]),
    }
    return {"ids": [r["id"] for r in keep], "y": y, "source": np.array([r["source"] for r in keep]),
            "utype": np.array([r["uncertainty_type"] for r in keep]),
            "s": {"s1": np.array([float(r["s1"]) for r in keep]), "s2": s2,
                  "s3": np.array([float(r["s3_heldout"]) for r in keep])}, "pred": pred}


def coverage_warning(all_rows, kept_ids):
    """The judged examples should look like the whole set. Returns a warning string or ''."""
    kept = [r for r in all_rows if r["id"] in kept_ids]
    if not kept:
        return "no examples in common"
    worst = 0.0
    shares = lambda rs, f: np.mean([f(r) for r in rs])  # noqa: E731
    worst = max(worst, abs(shares(kept, lambda r: r["should_refuse"] == "1") - shares(all_rows, lambda r: r["should_refuse"] == "1")))
    for t in {r["uncertainty_type"] for r in all_rows}:
        f = lambda r, t=t: r["uncertainty_type"] == t  # noqa: E731
        worst = max(worst, abs(shares(kept, f) - shares(all_rows, f)))
    if worst > 0.08:
        return (f"the judged examples are NOT a representative sample of the 1,600 (a share differs by "
                f"{worst:.0%}). The baseline was probably judged in ID order, so its first examples came from "
                "only a few providers. Do not trust this comparison until the baseline is fully judged.")
    return ""


# ------------------------------------------------------------------ statistics
def bootstrap(d, resamples, seed):
    y, src, pred = d["y"], d["source"], d["pred"]
    groups = {}
    for i, s in enumerate(src):
        groups.setdefault(s, []).append(i)
    keys = list(groups)
    garr = {k: np.array(v) for k, v in groups.items()}
    rng = np.random.default_rng(seed)
    point = {k: rates(y, p) for k, p in pred.items()}
    flat = {(k, m): [] for k in pred for m in METRICS}
    diff = {(k, m): [] for k in pred if k != "baseline" for m in METRICS}
    for _ in range(resamples):
        idx = np.concatenate([garr[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        yb = y[idx]
        if yb.all() or (~yb).all():
            continue
        r = {k: rates(yb, p[idx]) for k, p in pred.items()}
        for k in pred:
            for m in METRICS:
                flat[(k, m)].append(r[k][m])
                if k != "baseline":
                    diff[(k, m)].append(r[k][m] - r["baseline"][m])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    out = {"n": int(len(y)), "n_refuse_required": int(y.sum()), "n_answerable": int((~y).sum()),
           "n_sources": len(keys), "resamples": resamples, "rules": {}}
    for k in pred:
        out["rules"][k] = {m: {"value": point[k][m], "ci": ci(flat[(k, m)])} for m in METRICS}
        if k != "baseline":
            out["rules"][k]["diff"] = {m: {"value": point[k][m] - point["baseline"][m], "ci": ci(diff[(k, m)])}
                                       for m in METRICS}
    return out


def judgement(diff_bal):
    lo, hi = diff_bal["ci"]
    if lo > 0:
        return "BETTER"
    if hi < 0:
        return "WORSE"
    return "NO CLEAR DIFFERENCE"


# ------------------------------------------------------------------ plot
def plot(path, d, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = d["y"]
    colors = {"s1": BLUE, "s2": ORANGE, "s3": AQUA}
    names = {"s1": "S1 refuse-code probability", "s2": "S2 yes/no probability", "s3": "S3 hidden-state probe"}
    fig, ax = plt.subplots(figsize=(6.6, 5.1), facecolor=SURFACE)
    _style(ax)
    for k, s in d["s"].items():
        cand = np.concatenate([[s.min() - 1], np.unique(s), [s.max() + 1]])[::max(1, len(np.unique(s)) // 300)]
        ax.plot([(s[~y] > t).mean() for t in cand], [(s[y] <= t).mean() for t in cand], color=colors[k], linewidth=2,
                label=names[k])
    b = res["rules"]["baseline"]
    ax.plot(b["frr"]["value"], b["mrr"]["value"], "D", color=INK, markersize=10, markeredgecolor=SURFACE,
            markeredgewidth=2, label="baseline (judged replies)")
    ax.annotate("baseline", (b["frr"]["value"], b["mrr"]["value"]), textcoords="offset points", xytext=(8, 6),
                fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False refusal rate (answerable examples that were refused)", color=INK, fontsize=10)
    ax.set_ylabel("Missed refusal rate (refusal-required examples that were answered)", color=INK, fontsize=10)
    ax.set_title("Is the baseline inside or outside the curves? Lower left is better", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def _f(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def _ci(e):
    return f"{_f(e['value'])} [{_f(e['ci'][0])}, {_f(e['ci'][1])}]"


def by_type_table(d):
    """Missed and false refusal per uncertainty type, for the baseline, the S2 tuned threshold and the S3 probe,
    on the same examples as the main comparison. Point estimates only (each type has about 40 to 90 examples)."""
    rules = [("baseline", "Baseline"), ("s2_tuned", "S2 tuned"), ("s3_tuned", "S3 probe")]
    L = ["## By uncertainty type (missed refusal / false refusal)", "",
         "| Type | Must refuse / answerable | " + " | ".join(n for _, n in rules) + " |",
         "|---|---|" + "---|" * len(rules)]
    y, ut = d["y"], d["utype"]
    for t in sorted(set(ut)):
        m = ut == t
        cells = []
        for key, _ in rules:
            p = d["pred"][key][m]
            mrr = float((~p[y[m]]).mean()) if y[m].any() else float("nan")
            frr = float(p[~y[m]].mean()) if (~y[m]).any() else float("nan")
            cells.append(f"{mrr:.2f} / {frr:.2f}")
        L.append(f"| {t} | {int(y[m].sum())} / {int((~y[m]).sum())} | " + " | ".join(cells) + " |")
    L += ["", "Lower is better for both. Point estimates only: each type has about 40 to 90 examples.", ""]
    return L


def render(res, meta, warn, arrays=None):
    L = ["# Fair comparison: judged baseline vs Extension 2 (auto-generated)", ""]
    if warn:
        L += [f"> **WARNING: {warn}**", ""]
    L += [f"*Baseline run: `{meta['baseline_dir']}`. Baseline replies judged: {meta['n_judged']} of {meta['n_generated']} "
          f"generated ({meta['n_empty']} empty). Compared on {res['n']} examples ({res['n_refuse_required']} "
          f"refusal-required, {res['n_answerable']} answerable) from {res['n_sources']} source questions. "
          f"95% intervals: cluster bootstrap over source questions, {res['resamples']} resamples.*", ""]
    L += ["## Decisions side by side", "",
          "| Rule | False refusal | Missed refusal | Balanced accuracy | Detection F1 | Uses labels? |",
          "|---|---|---|---|---|---|"]
    b = res["rules"]["baseline"]
    L.append(f"| **Baseline (judged replies)** | {_ci(b['frr'])} | {_ci(b['mrr'])} | {_ci(b['bal_acc'])} | {_ci(b['f1'])} | no |")
    for k, label in RULES:
        r = res["rules"][k]
        uses = "no" if k.endswith("default") else "yes (cross-fitted)"
        L.append(f"| {label} | {_ci(r['frr'])} | {_ci(r['mrr'])} | {_ci(r['bal_acc'])} | {_ci(r['f1'])} | {uses} |")
    L += ["", "## Verdict per rule (balanced accuracy, rule minus baseline)", "",
          "| Rule | Difference [95% CI] | Verdict |", "|---|---|---|"]
    for k, label in RULES:
        d = res["rules"][k]["diff"]["bal_acc"]
        L.append(f"| {label} | {_ci(d)} | **{judgement(d)}** |")
    L += ["", "BETTER = the whole interval is above 0. WORSE = the whole interval is below 0. "
          "NO CLEAR DIFFERENCE = the interval includes 0.", ""]
    L += ["## Other differences (rule minus baseline)", "",
          "| Rule | False refusal | Missed refusal | Detection F1 |", "|---|---|---|---|"]
    for k, label in RULES:
        d = res["rules"][k]["diff"]
        L.append(f"| {label} | {_ci(d['frr'])} | {_ci(d['mrr'])} | {_ci(d['f1'])} |")
    L += ["", "Negative is good for false and missed refusal, positive is good for F1.", ""]
    if arrays is not None:
        L += by_type_table(arrays)
    L += ["## Read this before quoting a number", "",
          "- The baseline is one sample per example at temperature 1.0 (as in the paper), so it carries sampling noise "
          "that the interval only partly covers. Qwen was 4-bit.",
          "- The tuned thresholds and the S3 probe were fitted on the benchmark's labels (cross-fitted over source "
          "questions, so no example's own question was in its training data). The baseline gets no training. State this "
          "whenever the probe is compared with the baseline.",
          "- The two default-threshold rules use no labels: they are the cleanest comparison with the baseline.",
          "- The judge (Gemini) classifies free-text replies. A different judge could shift the baseline's numbers a little.",
          "- Detection only: this says nothing about answer correctness or about naming the right refusal reason.", "",
          "![trade-off](baseline_vs_scorers.png)", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", default=os.path.join(ROOT, "baseline", "results", "qwen15_7b_baseline_judged_day1.zip"),
                    help="a baseline run folder, or its zip (default: our committed graded run)")
    ap.add_argument("--novelty-dir", default=os.path.join(EXP, "results"))
    ap.add_argument("--subset-file", default=None, help="restrict to these example ids, e.g. data/novelty_sample_ids.json")
    ap.add_argument("--out", default=None, help="default: <novelty-dir>/comparison")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out or os.path.join(args.novelty_dir, "comparison")
    os.makedirs(out, exist_ok=True)
    score_rows = load_scores(args.novelty_dir)
    base_dec, n_gen, n_judged, n_empty = load_baseline(args.baseline_dir)
    subset = None
    if args.subset_file:
        path = args.subset_file if os.path.isabs(args.subset_file) else os.path.join(ROOT, args.subset_file)
        subset = set(json.load(open(path, encoding="utf-8"))["ids"])
    d = build_arrays(score_rows, base_dec, subset)
    if len(d["ids"]) < 50:
        raise SystemExit(f"Only {len(d['ids'])} examples are judged AND scored: judge more of the baseline first.")
    pool = [r for r in score_rows if subset is None or r["id"] in subset]
    warn = coverage_warning(pool, set(d["ids"]))
    if warn:
        print("WARNING:", warn)
    res = bootstrap(d, args.resamples, args.seed)
    meta = {"baseline_dir": args.baseline_dir, "n_generated": n_gen, "n_judged": n_judged, "n_empty": n_empty,
            "subset": args.subset_file}
    with open(os.path.join(out, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "warning": warn, "results": res}, f, indent=2, default=float)
    with open(os.path.join(out, "COMPARISON.md"), "w", encoding="utf-8") as f:
        f.write(render(res, meta, warn, arrays=d))
    try:
        plot(os.path.join(out, "baseline_vs_scorers.png"), d, res)
    except ImportError:
        print("matplotlib is not installed: the figure was skipped")
    print(open(os.path.join(out, "COMPARISON.md"), encoding="utf-8").read())
    print(f"written: {out}")


if __name__ == "__main__":
    main()
