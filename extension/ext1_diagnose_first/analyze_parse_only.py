"""Extension 1 headline numbers, from the committed replies alone: no model, no judge, no network.

A reply counts as a refusal only if it contains exactly one clean REFUSE_* code (the baseline's own parser).
Free-text refusals would need the judge, so the baseline and format-only refusal rates are LOWER BOUNDS.
For diagnose-first the evidence state is read directly from the text, so its numbers are exact.

All three conditions are compared on the examples they share (the 800 sample minus the baseline's
out-of-memory failures). Intervals: paired cluster bootstrap over source questions (variants of one
question are resampled together), 2,000 resamples, seed 0.

Usage, from the repository root:
    python extension/ext1_diagnose_first/analyze_parse_only.py
"""

import argparse
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline.refusalbench_repro.parser import parse_reply  # noqa: E402
from conditions import extract_effective  # noqa: E402

BASELINE_ZIP = ROOT / "baseline" / "results" / "qwen15_7b_baseline_judged_day1.zip"
EXT1_ZIP = ROOT / "extension" / "ext1_diagnose_first" / "results" / "ext1_runs.zip"
CONDITIONS = {"baseline": (BASELINE_ZIP, "qwen15_7b_baseline"),
              "format_only": (EXT1_ZIP, "qwen15_7b_format_only"),
              "category_first": (EXT1_ZIP, "qwen15_7b_category_first")}
NAMES = {"baseline": "Baseline", "format_only": "Format-only control", "category_first": "Diagnose first"}
# the source question is the third field of every example id: RB-NQ_<generator>_<source>_<type>_<intensity>_...
SOURCE_RE = re.compile(r"^RB-NQ_[a-z0-9]+_(-?\d+)_")


def read_replies(zip_path, folder):
    with zipfile.ZipFile(zip_path) as z, z.open(f"{folder}/target_outputs.jsonl") as f:
        return {r["example_id"]: r for r in map(json.loads, io.TextIOWrapper(f, encoding="utf-8")) if r}


def score(condition, row):
    """-> (refused with a clean code, the code or None, evidence state or None)"""
    eff = extract_effective(condition, row["raw_response"])
    status, code = parse_reply(eff["effective"])
    return status == "clean_code", code, eff["state"]


def rates(rows):
    must = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    return {"refusal_detected": np.mean([r["refused"] for r in must]),
            "correct_category": np.mean([r["code"] == r["expected"] for r in must]),
            "false_refusal": np.mean([r["refused"] for r in ans])}


def analyze(sample_file=ROOT / "data" / "novelty_sample_ids.json", B=2000, seed=0):
    sample = set(json.loads(Path(sample_file).read_text(encoding="utf-8"))["ids"])
    replies = {c: read_replies(*where) for c, where in CONDITIONS.items()}
    common = sorted(sample.intersection(*(set(r) for r in replies.values())))
    rows = {}
    for c, rep in replies.items():
        rows[c] = []
        for i in common:
            refused, code, state = score(c, rep[i])
            rows[c].append({"id": i, "source": SOURCE_RE.match(i).group(1), "expected": rep[i]["expected_behavior"],
                            "answerable": rep[i]["expected_behavior"] == "ANSWER_CORRECTLY",
                            "refused": refused, "code": code, "state": state})
    point = {c: rates(r) for c, r in rows.items()}

    # paired cluster bootstrap: the same resampled source questions for every condition
    sources = sorted({r["source"] for r in rows["baseline"]})
    idx = {c: {s: [r for r in rs if r["source"] == s] for s in sources} for c, rs in rows.items()}
    pairs = [("category_first", "format_only"), ("category_first", "baseline"), ("format_only", "baseline")]
    rng = np.random.default_rng(seed)
    draws = {p: {m: [] for m in point["baseline"]} for p in pairs}
    for _ in range(B):
        pick = rng.choice(len(sources), size=len(sources), replace=True)
        boot = {c: rates([r for k in pick for r in idx[c][sources[k]]]) for c in rows}
        for a, b in pairs:
            for m in boot[a]:
                draws[(a, b)][m].append(boot[a][m] - boot[b][m])
    diffs = {p: {m: (point[p[0]][m] - point[p[1]][m], *np.percentile(v, [2.5, 97.5])) for m, v in d.items()}
             for p, d in draws.items()}

    # the full 800 for the evidence states (diagnose-first has no out-of-memory gaps)
    cf = replies["category_first"]
    states = Counter(extract_effective("category_first", cf[i]["raw_response"])["state"] for i in sorted(sample))
    must_ids = [i for i in sample if cf[i]["expected_behavior"] != "ANSWER_CORRECTLY"]
    clear_on_must = sum(extract_effective("category_first", cf[i]["raw_response"])["state"] == "CLEAR"
                        for i in must_ids)
    return {"n_common": len(common), "n_sources": len(sources),
            "n_must_refuse": sum(not r["answerable"] for r in rows["baseline"]),
            "n_answerable": sum(r["answerable"] for r in rows["baseline"]),
            "point": point, "diffs": diffs, "states_800": dict(states),
            "clear_800": states.get("CLEAR", 0), "clear_on_must_refuse": (clear_on_must, len(must_ids))}


def render(res, B=2000):
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731
    out = ["# Extension 1: parse-only comparison", "",
           f"*Generated by `analyze_parse_only.py`. {res['n_common']} examples shared by all three conditions "
           f"({res['n_must_refuse']} must refuse, {res['n_answerable']} answerable) from {res['n_sources']} source "
           f"questions. A refusal counts only with one clean `REFUSE_*` code, so baseline and format-only rates are "
           f"lower bounds. Intervals: paired cluster bootstrap over source questions, {B} resamples, seed 0.*", "",
           "| Condition | Refusal detected | Correct refusal category | False refusal on answerable |",
           "|---|---:|---:|---:|"]
    for c, n in NAMES.items():
        p = res["point"][c]
        out.append(f"| {n} | {pct(p['refusal_detected'])} | {pct(p['correct_category'])} | {pct(p['false_refusal'])} |")
    out += ["", "| Difference (points) | Refusal detected | Correct refusal category | False refusal |", "|---|---:|---:|---:|"]
    for (a, b), d in res["diffs"].items():
        cells = [f"{pts(v)} [{pts(lo)}, {pts(hi)}]" for v, lo, hi in (d["refusal_detected"], d["correct_category"],
                                                                      d["false_refusal"])]
        out.append(f"| {NAMES[a]} minus {NAMES[b]} | " + " | ".join(cells) + " |")
    c, n = res["clear_on_must_refuse"]
    out += ["", f"Evidence states written by the model on all 800: {res['states_800']}.",
            f"It wrote CLEAR on {res['clear_800']} of 800, including {c} of the {n} examples that had to be refused."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "parse_only_comparison.md"))
    args = ap.parse_args()
    text = render(analyze(B=args.resamples), args.resamples)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
