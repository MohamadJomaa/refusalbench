# Baseline: RefusalBench-NQ with Qwen1.5-7B-Chat

Our reproduction of the RefusalBench evaluation (Muhamed et al., EACL 2026) for
one model, **Qwen1.5-7B-Chat**, on the released **RefusalBench-NQ** set (1,600
examples: 536 answerable, 1,064 that must be refused).

Nothing in this folder is our own idea. Our extensions are in `extension/`.

## How this folder was built

We took **path 2 of the template, a reimplementation that stays faithful to the
original code**. The authors' code
([github.com/aashiqmuhamed/refusalbench](https://github.com/aashiqmuhamed/refusalbench))
calls proprietary models through their APIs and a vLLM server on A100 GPUs,
neither of which fits the course's cost limit. We rewrote the runner for a free
Colab T4 and kept the parts that define the benchmark **verbatim**:

| Part | Source |
|---|---|
| Target-model prompt (`prompt_builder.py`) | copied verbatim from `naturalquestions/run_models_all.py`, `get_model_answer` |
| Judge prompt and its parser (`judge.py`) | copied verbatim from `classify_and_evaluate_response` (paper Appendix I.1.4) |
| Metrics (`metrics.py`) | the paper's definitions (Appendix D) and the original `run_models.py` |
| Dataset loader (`schema_adapter.py`) | maps the released Hugging Face fields; nothing is filtered or changed |
| Runner, resume, Gemini judge client (`runner.py`, `model_client.py`) | ours |

## What is in this folder

```
refusalbench_repro/       the code (a Python package, imported by the extensions)
  prompt_builder.py         the paper's prompt, verbatim
  judge.py                  the paper's judge prompt and parser, verbatim
  parser.py                 parse-first scoring: exactly one clean REFUSE_* code, or the judge
  metrics.py                answer and refusal accuracy, false and missed refusal, F1, CRS
  runner.py                 resumable generation and judging (append-only JSONL)
  model_client.py           Qwen in 4-bit (transformers + bitsandbytes), the Gemini judge, fakes for tests
  dataset_loader.py, schema_adapter.py, report.py
  conditions.py             plug-in point used by the extensions; only "baseline" is defined here
configs/qwen15_7b_baseline.yaml   model, precision, sampling, judge, stages
scripts/
  run_all.py                ONE command: generate (10, 100, all) -> judge -> metrics -> report
  run_smoke_test.py         dry run: prints the exact prompt for a 20-example sample, no model, no API
  judge_live_check.py       4 hand-made judge cases sent to the real Gemini judge (uses the API)
  check_gemini.py, probe_gemini.py   which Gemini models this key can use
  score_handcheck.py        agreement rate after a person fills in handcheck_sheet.csv
notebooks/
  colab_baseline.ipynb      full pipeline on a Colab T4 (generation + judging)
  colab_baseline_judge.ipynb  judging only, CPU runtime, any Google account
results/
  qwen15_7b_baseline_judged_day1.zip   the run: all replies + the graded ones, config, logs
  reproduction_report.md, metrics.json, metrics_by_*.csv, confusion_matrix.csv, handcheck_sheet.csv
```

## Commands

All commands run from the **repository root**.

**Check everything without a GPU or an API key** (fake model and fake judge,
20 examples, about 10 seconds):

```
python baseline/scripts/run_all.py --fake --dev-subset 20
```

**Print the exact prompt the model receives** (no model, no API):

```
python baseline/scripts/run_smoke_test.py --dry-run
```

**The real run** (needs a CUDA GPU with about 6 GB free; we used a free Colab T4
through `notebooks/colab_baseline.ipynb`):

```
python baseline/scripts/run_all.py --config baseline/configs/qwen15_7b_baseline.yaml --out runs
```

It generates 10 replies, checks them, then 100, then all 1,600, then judges and
writes the report to `runs/qwen15_7b_baseline/`. It is safe to stop and run
again: finished examples are never redone, and a settings lock refuses to mix
two configurations in one run folder.

**Judge saved replies only** (no GPU; set `GOOGLE_API_KEY` in `.env`, or the
Colab secret `GEMINI_API_KEY`):

```
python baseline/scripts/run_all.py --config baseline/configs/qwen15_7b_baseline.yaml --accept-partial --out runs
```

To continue from our saved run, unzip `results/qwen15_7b_baseline_judged_day1.zip`
into `runs/` first. On Colab, `notebooks/colab_baseline_judge.ipynb` does this from
Google Drive and stops by itself at the free daily quota.

## Our number next to the paper's

Qwen1.5-7B-Chat on RefusalBench-NQ. **770 of the 1,560 generated replies are
graded** (a seeded random sample: 261 answerable, 509 that must be refused).

| Metric | Paper | Ours (770 graded) |
|---|---:|---:|
| Answer accuracy (judge score 4 or 5) | **0.561** (text, Section 4.2, scaling) | **0.690** |
| Refusal accuracy (refused, right reason) | **about 0.05** (read from Figure 9, a small plot) | **0.114** |
| False refusal rate (answerable but refused) | not reported for this model | 0.165 |
| Missed refusal rate (had to refuse but answered) | not reported for this model | 0.456 |
| Refusal detection F1 | not reported for this model | 0.668 |
| Calibrated refusal score (mean of the two accuracies) | not reported for this model | 0.402 |

**What reproduces is the paper's finding, not its exact numbers.** In both the
paper and our run, Qwen1.5-7B almost never gives the correct reason for a
refusal, and it misses many refusals. Most missed: ambiguity (0.74) and
contradiction (0.71); most over-refused: false premise (false refusal 0.33). Our
answer accuracy is higher than the paper's. We do not read that as an
improvement, because the judge, the precision and the inference software all
differ (below).

**Reply format** (all 1,560): 528 clean refusal codes (34%) scored directly by
code; 991 free-text replies, 40 unknown codes and 1 reply with several codes
(1,032, 66%) sent to the judge. A reply with no code is never assumed to be an
answer, because a small model often refuses in plain words.

## Deviations from the paper

1. **Hardware and software:** one Colab T4 with Hugging Face transformers, not
   4 × A100 80 GB with vLLM. The chat template adds Qwen's default system prompt.
2. **Precision:** 4-bit (bitsandbytes). The paper does not state it; most likely
   16-bit. We did not measure the effect.
3. **Judge:** `gemini-3.5-flash-lite` instead of Claude-4-Sonnet, and only for
   replies without one clean refusal code. Same judge prompt, verbatim.
4. **Sampling:** temperature 1.0 and top_p 1.0, as in the paper. One seed (0), so
   numbers vary from run to run. `max_new_tokens` 256 (not stated in the paper).
5. **Coverage:** 40 examples (2.5%) failed with GPU out-of-memory and are
   excluded, not counted as errors; they are somewhat longer than average. The
   report lists them.

## What a full run would need

This is a smoke test in the course's sense. A full run needs:

| What | Size | Cost |
|---|---|---|
| Grade the remaining **790** replies | about 534 judge calls (the rest are scored by code) | $0: one more day of the free Gemini quota (500 calls per day), or a key from a second Google project |
| Regenerate the **40** out-of-memory examples | one by one on the T4 (about 6 s each), or a GPU with more memory | $0, a few minutes |
| **Hand-check 30** graded replies to validate the Gemini judge | fill in `results/handcheck_sheet.csv`, then `python baseline/scripts/score_handcheck.py baseline/results/handcheck_sheet.csv` | an hour of reading |
| A closer match to the paper | Claude-4-Sonnet as judge (about 1,000 calls of about 400 input and 100 output tokens), 16-bit weights on a GPU with at least 16 GB, several seeds | outside our free setup |

Measured speed: about **6 s per reply** on the T4 in 4-bit, so all 1,600 replies
take about 2.6 hours.

## Cost check

Generation ran on the free Colab T4. Judging used the free Gemini tier. Total
API spend: **$0**, inside the course's $20 limit.
