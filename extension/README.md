# Extension

Our own work on top of the baseline in `baseline/`. Nothing here changes the
baseline: both extensions call into `baseline/refusalbench_repro/` (the
diagnose-first prompts plug into the baseline runner through the `conditions.py`
hook; Extension 2 reuses the paper's prompt, the loader and the metrics), and
the baseline never imports extension code.

**Our finding in one line: decodable is not deployable.** Asking the model to
diagnose the evidence backfires (Extension 1); the information about when to
refuse is inside the model (Extension 2), but turning it into a clearly better
decision than the model's own is not shown.

| Folder | Idea | Result |
|---|---|---|
| [`ext1_diagnose_first/`](ext1_diagnose_first/README.md) | Ask the model to name what is wrong with the evidence before it answers | **Failed**: it wrote CLEAR on 762 of 800 and caught 1.9% of the needed refusals (baseline 43.3%) |
| [`ext2_reading_signals/`](ext2_reading_signals/README.md) | Do not ask the model: read its refusal probability, a yes/no probability, and a probe on its hidden state | **Signal, no proven gain**: AUROC 0.73 to 0.80, but the best rule (the probe) is +0.028 balanced accuracy over the baseline, interval [-0.014, +0.065] |

## The limitation both extensions target

RefusalBench shows that selective refusal is two skills, **detecting** that
the evidence is flawed and **naming why**, and that small models fail at both:
Qwen's refusal accuracy stays below 17% at every size (about 5% at 7B), and in
our baseline Qwen1.5-7B-Chat misses 45.6% of the refusals it should make and
wrongly refuses 16.5% of answerable questions. The paper measures this and
points to alignment training as the way forward; it does not propose a fix that
works on a small model **without training**. Both extensions keep the model's
weights fixed.

## How each is compared with the baseline

- **Same model and settings** as the baseline (Qwen1.5-7B-Chat, 4-bit,
  temperature 1.0, seed 0), on the same examples as the baseline run.
- **Paired comparison:** every condition is scored on exactly the same
  example IDs.
- **Uncertainty that respects the data:** the 1,600 examples are variants of
  only 100 source questions, so intervals come from a cluster bootstrap that
  resamples whole source questions.
- **A control** wherever a prompt changes, so a gain from tidier output is not
  mistaken for a gain from the idea.

## Smoke test, stated plainly

Both extensions are smoke tests in the course's sense: one model, one seed,
and a baseline graded on 770 of its 1,560 replies. What a full validation would
need is listed in each folder's README and in Section 5 of `report/REPORT.md`.
