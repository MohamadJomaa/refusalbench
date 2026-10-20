"""ONE entry point for Extension 2. Safe to stop and re-run: it resumes from the shards already on disk.

Order: load the 1,600 examples -> pre-flight check of the scorer -> score S1 (baseline prompt) -> score S2
(yes/no prompt) -> cross-fitted analysis -> RESULT_AUTO.md, plots, metrics.json.

Usage (the Colab notebook does this for you):
  python extension/ext2_reading_signals/run_ext2.py --out /content/drive/MyDrive/refusalbench-results
Test without a GPU:  python extension/ext2_reading_signals/run_ext2.py --fake --limit 40 --out runs
Tests only:  --fake (no GPU, no model), --limit N (a small stratified subset)
"""

import os

# less GPU memory fragmentation (must be set before torch is imported)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import json
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = HERE                                   # extension/ext2_reading_signals
ROOT = os.path.dirname(os.path.dirname(EXP))  # the repository root
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from analysis import run_analysis  # noqa: E402
from scorer import FakeScorer, HFScorer, load_shards, prompts_hash, run_scoring  # noqa: E402
from baseline.refusalbench_repro.dataset_loader import load_refusalbench_nq, select_smoke_sample  # noqa: E402
from baseline.refusalbench_repro.parser import parse_reply  # noqa: E402
from baseline.refusalbench_repro.runner import now, read_jsonl  # noqa: E402


def gpu_check(log):
    """Refuse to load the model when something else already holds the GPU (an older cell that is still alive)."""
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        used = int((total - free) / 2**20)
        log(f"GPU memory in use before loading the model: {used} MiB of {int(total / 2**20)} MiB")
        if used > 2000:
            raise SystemExit(f"STOPPED: {used} MiB of the GPU is already in use by another process. In Colab: "
                             "Runtime > Disconnect and delete runtime, reconnect, run again. Progress on Drive is kept.")
    except (ImportError, RuntimeError, AssertionError):
        pass


def assemble(examples, shard_dir):
    """Align the shard contents with the example list. Raises if anything is missing."""
    s1, s2 = load_shards(shard_dir, "s1"), load_shards(shard_dir, "s2")
    missing = [e["id"] for e in examples if e["id"] not in s1 or e["id"] not in s2]
    if missing:
        raise SystemExit(f"{len(missing)} examples are not scored yet (first: {missing[0]}). Run again to finish.")
    pick = lambda d, k: np.array([d[e["id"]][k] for e in examples])  # noqa: E731
    feats = {"s1_last": pick(s1, "hid_last").astype(np.float32), "s1_mid": pick(s1, "hid_mid").astype(np.float32),
             "s2_last": pick(s2, "hid_last").astype(np.float32), "s2_mid": pick(s2, "hid_mid").astype(np.float32)}
    return {
        "ids": [e["id"] for e in examples],
        "y": np.array([not e["is_answerable"] for e in examples]),
        "source": np.array([e["source_id"] for e in examples]),
        "utype": np.array([e["uncertainty_type"] for e in examples]),
        "intensity": np.array([e["intensity"] for e in examples]),
        "s1": pick(s1, "score").astype(np.float64), "s2": pick(s2, "score").astype(np.float64),
        "feats": feats,
    }, float(pick(s1, "mass").mean()), float(pick(s2, "mass").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(EXP, "configs", "ext2.yaml"))
    ap.add_argument("--out", default="runs", help="base folder; the run lives in <out>/<run_name>")
    ap.add_argument("--fake", action="store_true", help="TEST ONLY: synthetic scores, no GPU, no model")
    ap.add_argument("--limit", type=int, default=0, help="TEST ONLY: a small stratified subset")
    ap.add_argument("--analysis-only", action="store_true", help="skip scoring, analyse the shards that exist")
    ap.add_argument("--baseline-dir", default=None, help="a baseline run folder, for the reference line")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.fake:
        cfg = copy.deepcopy(cfg)
        cfg["run_name"] += "_FAKE_TEST"
    run_dir = os.path.join(args.out, cfg["run_name"])
    shard_dir = os.path.join(run_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{now()}  {msg}\n")

    log(f"=== Extension 2: confidence-threshold refusal gate | run folder: {run_dir}")

    # ---- 1. data: every example, longest context first so an out-of-memory error shows up at the start
    examples = load_refusalbench_nq()
    if len(examples) != 1600 and not args.limit:
        raise SystemExit(f"Dataset is not the expected 1,600 examples: {len(examples)}")
    if args.limit:
        log(f"!!! TEST SUBSET of {args.limit}: not a real run")
        examples = select_smoke_sample(examples, n=args.limit, seed=0)
    examples = sorted(examples, key=lambda e: (-len(e["context"]), e["id"]))
    log(f"{len(examples)} examples, {sum(e['is_answerable'] for e in examples)} answerable, "
        f"{len({e['source_id'] for e in examples})} source questions")

    # ---- 2. lock: refuse to mix settings inside one run folder
    lock = {"model": cfg["target"]["model"], "precision": cfg["target"]["precision"], "seed": cfg["target"]["seed"],
            "prompts_hash": prompts_hash(), "limit": args.limit, "fake": bool(args.fake)}
    lock_path = os.path.join(run_dir, "config.json")
    if os.path.exists(lock_path):
        if json.load(open(lock_path, encoding="utf-8")) != lock:
            raise SystemExit("This run folder was started with different settings. Change run_name to start fresh.")
    else:
        json.dump(lock, open(lock_path, "w", encoding="utf-8"), indent=2)
    with open(os.path.join(run_dir, "config_full.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # ---- 3. scoring
    s1_mode = cfg["scoring"]["s1_mode"]
    mode_path = os.path.join(run_dir, "s1_mode.json")
    if not args.analysis_only:
        have = [len(load_shards(shard_dir, k)) for k in ("s1", "s2")] if os.listdir(shard_dir) else [0, 0]
        if have[0] >= len(examples) and have[1] >= len(examples):
            log("all examples are already scored, skipping the model")
            scorer = None
        else:
            if args.fake:
                scorer = FakeScorer(cfg)
            else:
                log(f"loading {cfg['target']['model']} ({cfg['target']['precision']}) ...")
                gpu_check(log)
                scorer = HFScorer({**cfg["target"], "s1_mode": s1_mode}, log=log)
            if os.path.exists(mode_path):
                scorer.s1_mode = json.load(open(mode_path))["s1_mode"]  # never mix modes across a resume
                log(f"resuming with S1 mode {scorer.s1_mode!r}")
            else:
                scorer.s1_mode = s1_mode
                scorer.s1_mode = scorer.preflight(examples)
                json.dump({"s1_mode": scorer.s1_mode}, open(mode_path, "w"))
        if scorer is not None:
            bs, ss = cfg["target"]["batch_size"], cfg["scoring"]["shard_size"]
            for kind in ("s1", "s2"):
                run_scoring(examples, scorer, shard_dir, kind, ss, bs, log)
    if os.path.exists(mode_path):
        s1_mode = json.load(open(mode_path))["s1_mode"]

    # ---- 4. analysis
    data, mass1, mass2 = assemble(examples, shard_dir)
    novelty_ids = set(json.load(open(os.path.join(ROOT, cfg["subset_file"]), encoding="utf-8"))["ids"])
    mask = np.array([i in novelty_ids for i in data["ids"]])
    baseline_pred = None
    bdir = args.baseline_dir or cfg.get("baseline_dir")
    if bdir and os.path.exists(os.path.join(bdir, "target_outputs.jsonl")):
        rows = read_jsonl(os.path.join(bdir, "target_outputs.jsonl"))
        baseline_pred = {r["example_id"]: int(parse_reply(r["raw_response"])[0] == "clean_code") for r in rows}
        log(f"baseline reference: {len(baseline_pred)} replies from {bdir}")
    meta = {"model": cfg["target"]["model"], "precision": cfg["target"]["precision"], "s1_mode": s1_mode,
            "mean_p_refuse_s1": mass1, "mean_yes_no_mass_s2": mass2, "run_dir": run_dir, "fake": bool(args.fake),
            "prompts_hash": prompts_hash(), "n_novelty_subset": int(mask.sum())}
    res = run_analysis(data, run_dir, cfg["analysis"], meta, novelty_mask=mask, baseline_pred=baseline_pred, log=log)
    from analysis import verdict

    v, h1, h2 = verdict(res["all"])
    log("=== DONE")
    log(f"S1 AUROC {res['all']['auroc']['s1']['value']:.3f} | S2 {res['all']['auroc']['s2']['value']:.3f} | "
        f"S3 {res['all']['auroc']['s3']['value']:.3f}")
    log(f"verdict (mechanical): {v}")
    log(f"report: {os.path.join(run_dir, 'RESULT_AUTO.md')}")
    print("\n" + open(os.path.join(run_dir, "RESULT_AUTO.md"), encoding="utf-8").read()[:3500])


if __name__ == "__main__":
    main()
