"""Analysis for Extension 2. Judge-free: every number comes from the benchmark's own labels.

Positive class = "should refuse" (expected behavior is not ANSWER_CORRECTLY). A scorer gives one number per
example, higher = refuse; a decision is "refuse iff score > threshold".

  false refusal rate  (FRR) = answerable examples that get refused
  missed refusal rate (MRR) = refusal-required examples that get answered
  balanced accuracy         = 1 - (FRR + MRR) / 2        detection F1 = F1 of the "refuse" class

No leakage: all 100 source questions appear in both the dev pool and the novelty 800, so a threshold or probe
fitted on one pool would see the same questions in the other. Instead everything is CROSS-FITTED over source
questions: examples of one source question are never in the data used to choose their own threshold or fit their
own probe. Held-out predictions are pooled and evaluated, with a cluster bootstrap over source questions.
"""

import csv
import json
import os
import warnings

import numpy as np
from scipy.stats import rankdata

SCORERS = ("s1", "s2", "s3")
NAMES = {"s1": "S1 refuse-code probability (the novelty)", "s2": "S2 yes/no sufficiency probability",
         "s3": "S3 hidden-state linear probe"}
SHORT = {"s1": "S1", "s2": "S2", "s3": "S3"}


# ------------------------------------------------------------------ metrics
def auroc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.asarray(y).astype(int), s))


def rates(y, pred):
    """y, pred: 1 = refuse. Returns the four headline numbers."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    ans, ref = ~y, y
    frr = float(pred[ans].mean()) if ans.any() else float("nan")
    mrr = float((~pred[ref]).mean()) if ref.any() else float("nan")
    tp, fp, fn = (pred & ref).sum(), (pred & ans).sum(), (~pred & ref).sum()
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    return {"frr": frr, "mrr": mrr, "bal_acc": 1 - (frr + mrr) / 2, "f1": f1}


def pick_threshold(y, s, criterion="balanced_accuracy", frr_cap=0.10, max_candidates=400):
    """Choose the threshold on TRAINING data only. Returns a float."""
    y, s = np.asarray(y).astype(bool), np.asarray(s, dtype=np.float64)
    u = np.unique(s)
    if len(u) > max_candidates:
        u = np.quantile(s, np.linspace(0, 1, max_candidates))
        u = np.unique(u)
    cand = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    pred = s[:, None] > cand[None, :]
    ans, ref = ~y, y
    frr = pred[ans].mean(0) if ans.any() else np.zeros(len(cand))
    mrr = (~pred[ref]).mean(0) if ref.any() else np.zeros(len(cand))
    tp = (pred & ref[:, None]).sum(0)
    fp = (pred & ans[:, None]).sum(0)
    fn = (~pred & ref[:, None]).sum(0)
    if criterion == "balanced_accuracy":
        val = 1 - (frr + mrr) / 2
    elif criterion == "f1":
        val = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    elif criterion == "frr_cap":  # fewest missed refusals while keeping false refusals under the cap
        val = np.where(frr <= frr_cap, 1 - mrr, -1.0)
    else:
        raise ValueError(criterion)
    best = np.flatnonzero(val >= val.max() - 1e-12)
    return float(cand[best[len(best) // 2]])  # middle of the tied thresholds, for stability


def make_folds(sources, k, seed):
    """Assign every source question to one of k folds. Returns an int array aligned with `sources`."""
    uniq = sorted(set(sources))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {uniq[i]: int(rank % k) for rank, i in enumerate(perm)}
    return np.array([fold_of[s] for s in sources])


# ------------------------------------------------------------------ the probe (S3)
def _probe(C, n_comp, seed):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), PCA(n_components=n_comp, random_state=seed),
                         LogisticRegression(C=C, max_iter=500, class_weight="balanced"))


def _n_comp(X, n_comp):
    return int(max(2, min(n_comp, X.shape[1], X.shape[0] - 1)))


def inner_oof(X, y, groups, C, n_comp, seed, splits=3):
    """Out-of-fold decision scores inside the training data, grouped by source question."""
    from sklearn.model_selection import GroupKFold

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=splits).split(X, y, groups):
        m = _probe(C, _n_comp(X[tr], n_comp), seed).fit(X[tr], y[tr])
        oof[te] = m.decision_function(X[te])
    return oof


def fit_probe_on_train(feats, y, groups, cfg):
    """Choose feature set and C by inner grouped CV (AUROC), refit on all training rows.
    Returns (predict_fn, oof_scores_of_the_chosen_config, chosen_name, chosen_C)."""
    best = None
    for name, X in feats.items():
        for C in cfg["c_grid"]:
            oof = inner_oof(X, y, groups, C, cfg["pca_components"], cfg["seed"])
            a = auroc(y, oof)
            if best is None or a > best[0]:
                best = (a, name, C, oof)
    _, name, C, oof = best
    X = feats[name]
    model = _probe(C, _n_comp(X, cfg["pca_components"]), cfg["seed"]).fit(X, y)
    return (lambda Xnew_by_name: model.decision_function(Xnew_by_name[name])), oof, name, C


# ------------------------------------------------------------------ cross-fitting
def crossfit(data, cfg, log=print):
    """Held-out score and decision for every example, using only other source questions to tune anything."""
    y, src = data["y"], data["source"]
    n = len(y)
    fold = make_folds(src, cfg["folds"], cfg["seed"])
    out = {"fold": fold, "s3": np.zeros(n)}
    pred = {k: np.zeros(n, dtype=bool) for k in ("s1_default", "s1_tuned", "s2_default", "s2_tuned", "s3_tuned")}
    thr = {k: [] for k in ("s1", "s2", "s3")}
    chosen = []
    for k in range(cfg["folds"]):
        te, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        for s in ("s1", "s2"):
            t = pick_threshold(y[tr], data[s][tr], cfg["criterion"])
            thr[s].append(t)
            pred[f"{s}_tuned"][te] = data[s][te] > t
        pred["s1_default"][te] = data["s1"][te] > 0.0  # p(refuse) > 0.5: what Qwen would do on its own
        pred["s2_default"][te] = data["s2"][te] > 0.0
        feats_tr = {nm: X[tr] for nm, X in data["feats"].items()}
        predict, oof, name, C = fit_probe_on_train(feats_tr, y[tr].astype(int), src[tr], cfg)
        t3 = pick_threshold(y[tr], oof, cfg["criterion"])
        thr["s3"].append(t3)
        out["s3"][te] = predict({nm: X[te] for nm, X in data["feats"].items()})
        pred["s3_tuned"][te] = out["s3"][te] > t3
        chosen.append({"fold": k, "features": name, "C": C})
        log(f"  fold {k}: train {len(tr)}, held out {len(te)}; probe uses {name}, C={C}")
    return out, pred, thr, chosen


# ------------------------------------------------------------------ evaluation with a cluster bootstrap
def _point(y, scores, pred):
    res = {"auroc": {k: auroc(y, v) for k, v in scores.items()}}
    res["ops"] = {k: rates(y, p) for k, p in pred.items()}
    return res


def _diffs(pt):
    o, a = pt["ops"], pt["auroc"]
    return {
        "bal_acc: S1 tuned - S1 default": o["s1_tuned"]["bal_acc"] - o["s1_default"]["bal_acc"],
        "f1: S1 tuned - S1 default": o["s1_tuned"]["f1"] - o["s1_default"]["f1"],
        "AUROC: S1 - S2": a["s1"] - a["s2"],
        "AUROC: S3 - S1": a["s3"] - a["s1"],
        "bal_acc: S3 tuned - S1 tuned": o["s3_tuned"]["bal_acc"] - o["s1_tuned"]["bal_acc"],
    }


def evaluate(y, source, scores, pred, resamples, seed):
    """Point estimates plus 95% cluster-bootstrap intervals (resampling whole source questions)."""
    pt = _point(y, scores, pred)
    d_pt = _diffs(pt)
    groups = {}
    for i, s in enumerate(source):
        groups.setdefault(s, []).append(i)
    keys = list(groups)
    garr = {k: np.array(v) for k, v in groups.items()}
    rng = np.random.default_rng(seed)
    flat_auc = {k: [] for k in scores}
    flat_ops = {(o, m): [] for o in pred for m in ("frr", "mrr", "bal_acc", "f1")}
    flat_d = {k: [] for k in d_pt}
    for _ in range(resamples):
        idx = np.concatenate([garr[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        yb = y[idx]
        if yb.all() or (~yb).all():
            continue
        b = _point(yb, {k: v[idx] for k, v in scores.items()}, {k: v[idx] for k, v in pred.items()})
        for k in scores:
            flat_auc[k].append(b["auroc"][k])
        for (o, m) in flat_ops:
            flat_ops[(o, m)].append(b["ops"][o][m])
        for k, v in _diffs(b).items():
            flat_d[k].append(v)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    return {
        "n": int(len(y)), "n_refuse_required": int(y.sum()), "n_answerable": int((~y).sum()),
        "n_sources": len(keys), "resamples": resamples,
        "auroc": {k: {"value": pt["auroc"][k], "ci": ci(flat_auc[k])} for k in scores},
        "ops": {o: {m: {"value": pt["ops"][o][m], "ci": ci(flat_ops[(o, m)])}
                    for m in ("frr", "mrr", "bal_acc", "f1")} for o in pred},
        "diffs": {k: {"value": d_pt[k], "ci": ci(flat_d[k])} for k in d_pt},
    }


def verdict(ev):
    """Mechanical reading of the two pre-registered claims (see README)."""
    h1 = ev["auroc"]["s1"]["ci"][0] > 0.5
    h2 = ev["diffs"]["bal_acc: S1 tuned - S1 default"]["ci"][0] > 0.0
    if h1 and h2:
        return "SUPPORTED", h1, h2
    if h1:
        return "SIGNAL BUT NO GAIN FROM THE THRESHOLD", h1, h2
    return "NOT SUPPORTED", h1, h2


# ------------------------------------------------------------------ plots
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e5e4df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLOR = {"s1": BLUE, "s2": ORANGE, "s3": AQUA}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_tradeoff(path, y, scores, pred):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.0), facecolor=SURFACE)
    _style(ax)
    for k, s in scores.items():
        s = np.asarray(s)
        cand = np.concatenate([[s.min() - 1], np.unique(s), [s.max() + 1]])
        frr = np.array([(s[~y] > t).mean() for t in cand[::max(1, len(cand) // 300)]])
        mrr = np.array([(s[y] <= t).mean() for t in cand[::max(1, len(cand) // 300)]])
        ax.plot(frr, mrr, color=COLOR[k], linewidth=2, label=NAMES[k])
    for k, color in (("s1", BLUE), ("s2", ORANGE), ("s3", AQUA)):
        key = f"{k}_tuned"
        r = rates(y, pred[key])
        ax.plot(r["frr"], r["mrr"], "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    for k in ("s1", "s2"):
        r = rates(y, pred[f"{k}_default"])
        ax.plot(r["frr"], r["mrr"], "o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=COLOR[k],
                markeredgewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False refusal rate (answerable examples that were refused)", color=INK, fontsize=10)
    ax.set_ylabel("Missed refusal rate (refusal-required examples that were answered)", color=INK, fontsize=10)
    ax.set_title("Answer-vs-refuse trade-off: lower left is better", color=INK, fontsize=11, loc="left")
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=9),
                Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED,
                       markeredgewidth=2, markersize=9)]
    labels += ["threshold chosen on other source questions", "default threshold (p > 0.5, No > Yes)"]
    ax.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_s1_distribution(path, y, s1, thresholds):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s1 = np.asarray(s1, dtype=float)
    lo, hi = min(np.percentile(s1, 0.5), -3.0), max(np.percentile(s1, 99.5), 3.0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    s1c = np.clip(s1, lo, hi)
    bins = np.linspace(lo, hi, 41)
    fig, ax = plt.subplots(figsize=(6.4, 3.9), facecolor=SURFACE)
    _style(ax)
    ax.hist(s1c[~y], bins=bins, histtype="step", linewidth=2, color=BLUE, label="should answer")
    ax.hist(s1c[y], bins=bins, histtype="step", linewidth=2, color=ORANGE, label="should refuse")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.18)
    ax.axvline(0, color=MUTED, linewidth=1.2, linestyle="--")
    marks = [(0.0, "default (p = 0.5)", MUTED)]
    if thresholds:
        t = float(np.median(thresholds))
        ax.axvline(t, color=INK, linewidth=1.5)
        marks.append((t, "tuned (median of folds)", INK))
    marks.sort()  # the left line is labelled to its left, the right line to its right, so labels never overlap
    for i, (x, label, color) in enumerate(marks):
        left = i == 0 and len(marks) > 1
        ax.text(x, top * 1.14, (label + " ") if left else (" " + label), color=color, fontsize=8, va="top",
                ha="right" if left else "left")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("S1 score: log-odds that Qwen opens with a refusal code", color=INK, fontsize=10)
    ax.set_ylabel("examples", color=INK, fontsize=10)
    ax.set_title("Does Qwen's refusal probability separate the two groups?", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center left", bbox_to_anchor=(0.0, 0.72))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ report
def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _ci(e, d=3):
    return f"{_f(e['value'], d)} [{_f(e['ci'][0], d)}, {_f(e['ci'][1], d)}]"


def _ops_table(ev):
    rows = [("S1 default (p > 0.5)", "s1_default"), ("S1 tuned threshold", "s1_tuned"),
            ("S2 default (No > Yes)", "s2_default"), ("S2 tuned threshold", "s2_tuned"),
            ("S3 probe (tuned)", "s3_tuned")]
    out = ["| Decision rule | False refusal | Missed refusal | Balanced accuracy | Detection F1 |", "|---|---|---|---|---|"]
    for label, k in rows:
        o = ev["ops"][k]
        out.append(f"| {label} | {_ci(o['frr'])} | {_ci(o['mrr'])} | {_ci(o['bal_acc'])} | {_ci(o['f1'])} |")
    return "\n".join(out)


def render_report(res, cfg, meta):
    ev = res["all"]
    v, h1, h2 = verdict(ev)
    ev8 = res.get("novelty800")
    L = []
    L.append("# Extension 2 results (auto-generated): confidence-threshold refusal gate")
    L.append("")
    L.append(f"*Generated automatically from `{meta['run_dir']}`. Read it together with `README.md` before writing "
             "the final `RESULT.md`.*" + ("  \n**THIS RUN USED A FAKE SCORER: the numbers are meaningless.**" if meta.get("fake") else ""))
    L.append("")
    L.append(f"## Verdict (mechanical, rule fixed in the README before the run): **{v}**")
    L.append("")
    L.append(f"- Signal check, S1 AUROC lower bound above 0.5: **{'yes' if h1 else 'no'}** "
             f"(AUROC {_ci(ev['auroc']['s1'])})")
    d = ev["diffs"]["bal_acc: S1 tuned - S1 default"]
    L.append(f"- Gate check, tuned threshold beats the default in balanced accuracy with the interval above 0: "
             f"**{'yes' if h2 else 'no'}** (difference {_ci(d)})")
    L.append("")
    L.append(f"## Data and protocol")
    L.append("")
    L.append(f"- {ev['n']} examples ({ev['n_refuse_required']} refusal-required, {ev['n_answerable']} answerable) from "
             f"{ev['n_sources']} source questions. Model `{meta['model']}` ({meta['precision']}), one forward pass per "
             f"example, no generation, no judge.")
    L.append(f"- Thresholds and the probe are **cross-fitted** over source questions ({cfg['folds']} folds, "
             f"criterion `{cfg['criterion']}`): an example's own source question is never in the data used to tune "
             f"its decision rule. Intervals are 95% cluster bootstrap over source questions ({ev['resamples']} resamples).")
    L.append(f"- S1 mode: `{meta['s1_mode']}`. Mean probability that Qwen opens with a refusal code: "
             f"{_f(meta['mean_p_refuse_s1'])}. Mean probability mass on the words Yes/No in S2: "
             f"{_f(meta['mean_yes_no_mass_s2'])} (close to 1 means Qwen answers the yes/no question as asked).")
    L.append("")
    L.append("## Ranking quality (threshold-free)")
    L.append("")
    L.append("| Scorer | AUROC [95% CI] | AUPRC |")
    L.append("|---|---|---|")
    for k in SCORERS:
        L.append(f"| {NAMES[k]} | {_ci(ev['auroc'][k])} | {_f(res['auprc'][k])} |")
    L.append("")
    L.append("0.5 is chance, 1.0 is perfect.")
    L.append("")
    L.append("## Decisions on all examples")
    L.append("")
    L.append(_ops_table(ev))
    L.append("")
    L.append("Lower is better for false and missed refusal; higher is better for balanced accuracy and F1.")
    L.append("")
    if ev8:
        L.append(f"### Same table on the {ev8['n']} novelty examples only (same predictions, subset)")
        L.append("")
        L.append(_ops_table(ev8))
        L.append("")
    L.append("## Differences (paired, same examples)")
    L.append("")
    L.append("| Comparison | Difference [95% CI] |")
    L.append("|---|---|")
    for k, e in ev["diffs"].items():
        L.append(f"| {k} | {_ci(e)} |")
    L.append("")
    if res.get("baseline_ref"):
        b = res["baseline_ref"]
        L.append("## Reference: the sampled baseline replies")
        L.append("")
        L.append(f"On the {b['n']} examples present in both: clean `REFUSE_*` code on {_f(1 - b['mrr'])} of the "
                 f"refusal-required examples (missed refusal {_f(b['mrr'])}) and on {_f(b['frr'])} of the answerable "
                 "ones (false refusal). **This is a lower bound for the baseline**: free-text refusals without a code "
                 "are counted as answers because no judge was used. Treat it as context, not as the main comparison.")
        L.append("")
    L.append("## By uncertainty type (S1 tuned, all examples)")
    L.append("")
    L.append("| Type | refusal-required | missed refusal | answerable | false refusal |")
    L.append("|---|---|---|---|---|")
    for t, r in res["by_type"].items():
        L.append(f"| {t} | {r['n_ref']} | {_f(r['mrr'])} | {r['n_ans']} | {_f(r['frr'])} |")
    L.append("")
    L.append("## Probe configuration chosen in each fold")
    L.append("")
    for c in res["probe_choice"]:
        L.append(f"- fold {c['fold']}: features `{c['features']}`, C = {c['C']}")
    L.append("")
    L.append("## Figures")
    L.append("")
    L.append("- `tradeoff.png`: false vs missed refusal for every threshold.")
    L.append("- `s1_distribution.png`: S1 score for the two groups.")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ driver
def run_analysis(data, out_dir, cfg, meta, novelty_mask=None, baseline_pred=None, log=print):
    """data: ids, y (bool), source, utype, intensity (arrays), s1, s2 (arrays), feats (dict of arrays)."""
    os.makedirs(out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")
    y, src = data["y"], data["source"]
    log("--- cross-fitting thresholds and the probe over source questions")
    cf, pred, thr, chosen = crossfit(data, cfg, log=log)
    scores = {"s1": data["s1"], "s2": data["s2"], "s3": cf["s3"]}
    log("--- bootstrap")
    res = {"all": evaluate(y, src, scores, pred, cfg["bootstrap"], cfg["seed"]),
           "auprc": {k: auprc(y, scores[k]) for k in scores},
           "thresholds": {k: [float(t) for t in v] for k, v in thr.items()}, "probe_choice": chosen}
    if novelty_mask is not None and novelty_mask.sum() > 0:
        m = novelty_mask
        res["novelty800"] = evaluate(y[m], src[m], {k: v[m] for k, v in scores.items()},
                                     {k: v[m] for k, v in pred.items()}, cfg["bootstrap"], cfg["seed"] + 1)
    by_type = {}
    for t in sorted(set(data["utype"])):
        m = data["utype"] == t
        ref, ans = m & y, m & ~y
        by_type[t] = {"n_ref": int(ref.sum()), "n_ans": int(ans.sum()),
                      "mrr": float((~pred["s1_tuned"][ref]).mean()) if ref.any() else float("nan"),
                      "frr": float(pred["s1_tuned"][ans].mean()) if ans.any() else float("nan")}
    res["by_type"] = by_type
    if baseline_pred:
        ids = [i for i, x in enumerate(data["ids"]) if x in baseline_pred]
        if ids:
            ids = np.array(ids)
            bp = np.array([baseline_pred[data["ids"][i]] for i in ids], dtype=bool)
            r = rates(y[ids], bp)
            res["baseline_ref"] = {"n": int(len(ids)), "frr": r["frr"], "mrr": r["mrr"]}
    meta = dict(meta)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": res}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "uncertainty_type", "intensity", "should_refuse", "fold", "s1", "s2", "s3_heldout",
                    "s1_default", "s1_tuned", "s2_tuned", "s3_tuned"])
        for i in range(len(y)):
            w.writerow([data["ids"][i], src[i], data["utype"][i], data["intensity"][i], int(y[i]), int(cf["fold"][i]),
                        f"{data['s1'][i]:.4f}", f"{data['s2'][i]:.4f}", f"{cf['s3'][i]:.4f}", int(pred["s1_default"][i]),
                        int(pred["s1_tuned"][i]), int(pred["s2_tuned"][i]), int(pred["s3_tuned"][i])])
    plot_tradeoff(os.path.join(out_dir, "tradeoff.png"), y, scores, pred)
    plot_s1_distribution(os.path.join(out_dir, "s1_distribution.png"), y, data["s1"], thr["s1"])
    with open(os.path.join(out_dir, "RESULT_AUTO.md"), "w", encoding="utf-8") as f:
        f.write(render_report(res, cfg, meta))
    log(f"written: {os.path.join(out_dir, 'RESULT_AUTO.md')}")
    return res
