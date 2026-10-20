# Extension 1: diagnose the evidence first (negative result)

**Verdict: it failed, clearly.** Asked to name what is wrong with the evidence
before answering, Qwen1.5-7B-Chat wrote `EVIDENCE_STATE: CLEAR` on **762 of 800**
examples, including **504 of the 532** that had to be refused, and then answered.
It caught **1.9%** of the needed refusals, against **43.3%** for the baseline.

## The limitation it targets

RefusalBench shows that refusing well takes two separate skills: **detecting**
that the evidence is flawed, and **naming why** (Section 4.2, Figure 6). Small
models are weakest at the second: Qwen's refusal accuracy stays below 17% at
every size, about 5% at 7B, and in our baseline Qwen1.5-7B also misses 45.6% of
the refusals it should make. The paper measures this weakness and points to
targeted alignment training; it does not offer a fix that works without
training.

**Hypothesis:** if the model must first write down which of the six problems
(or none) the evidence has, and the refusal follows from that diagnosis, its
refusal decisions improve. One model call per example, no training, the
benchmark's own definitions.

## What we changed relative to the baseline

Only the end of the prompt. Everything else is the baseline: same model,
4-bit, temperature 1.0, top_p 1.0, seed 0, max 256 new tokens, the same
definitions of the six refusal codes (a test checks the text is identical).

| Condition | What the model is told, on top of the paper's prompt | Purpose |
|---|---|---|
| Baseline | nothing: answer, or reply with only a `REFUSE_*` code | the paper's method |
| **Format-only control** | end with one line `FINAL: <answer or code>` | separates "tidier output" from "better reasoning" |
| **Diagnose first** | write `EVIDENCE_STATE: <CLEAR or one of the six problems>`, then `FINAL:` | our idea |

How a diagnose-first reply is read (`conditions.py`, `extract_effective`): a
non-CLEAR state **decides** the outcome and maps to its refusal code (for
example `CONTRADICTORY` to `REFUSE_CONTRADICTORY_CONTEXT`); for CLEAR, the text
after `FINAL:` is the answer. Disagreements between the state and the FINAL line
are flagged, never silently resolved.

The headline comparison is **diagnose first against the format-only control**:
if diagnose first had beaten the baseline but not the control, the gain would
have come from the output format, not from the diagnosis.

## How it is evaluated against the baseline

Same 800 examples for all three conditions (`data/novelty_sample_ids.json`: 268
answerable, 532 to refuse, all 100 source questions). The comparison uses the
**776** examples all three share (the baseline lost 24 of the 800 to GPU
out-of-memory): 517 to refuse, 259 answerable, 99 source questions.

Scoring needs **no judge**: a refusal counts only when the reply has exactly
one clean `REFUSE_*` code (the baseline's own parser). So the baseline and
control rates are **lower bounds** (a free-text refusal would need the judge);
diagnose first's are exact, because the state is read from the text. Intervals
are a paired cluster bootstrap over source questions (2,000 resamples, seed 0):
variants of one question are resampled together, because they are not
independent.

## Results

From `results/parse_only_comparison.md`, produced by `analyze_parse_only.py`:

| Condition | Refusal detected | Correct refusal category | False refusal on answerable |
|---|---:|---:|---:|
| Baseline | 43.3% | 6.8% | 15.4% |
| Format-only control | 35.8% | 3.3% | 7.7% |
| **Diagnose first** | **1.9%** | **0.0%** | 0.4% |

| Difference, in points [95% interval] | Refusal detected | Correct category |
|---|---:|---:|
| Diagnose first minus control | -33.8 [-38.2, -29.5] | -3.3 [-4.8, -2.0] |
| Diagnose first minus baseline | -41.4 [-46.1, -36.7] | -6.8 [-9.2, -4.6] |
| Control minus baseline | -7.5 [-11.3, -3.9] | -3.5 [-5.9, -1.3] |

Every interval excludes zero. The control is also slightly worse than the
baseline, so the format alone does not explain the collapse.

**What the model wrote.** CLEAR on 762 of 800. A valid problem state on only
**7** (all `MISSING_INFORMATION`). Malformed states on 23 (`CLARITY`, `CLEARS`,
`_CLEAR`, ...) and none at all on 8. By type, CLEAR on: ambiguity 68 of 69,
contradiction 87 of 94, missing information 87 of 92, false premise 89 of 93,
granularity 83 of 92, epistemic 90 of 92, answerable 258 of 268. Even if every
non-CLEAR reply had been a perfect refusal, the ceiling was 38 of 800 (about 5%).

**Why it failed: we do not know.** The model may anchor on CLEAR, which is the
first option listed; that is a guess, not a finding, because we tested nothing
about the mechanism. Testing it is the first item below.

## Files

| File | What it is |
|---|---|
| `conditions.py` | The two prompts and the reply extraction. Loaded by `baseline/scripts/run_all.py` as a plug-in (`experiment_dir` in the config); the baseline code never imports it |
| `configs/qwen15_7b_format_only.yaml`, `configs/qwen15_7b_category_first.yaml` | One config per condition; identical to the baseline config except `condition`, `run_name`, `subset_file`, `experiment_dir` |
| `analyze_parse_only.py` | The headline numbers above, from the committed replies only (no model, no judge, no network) |
| `compare.py`, `compare_conditions.py` | The same comparison on **judged** replies, with answer accuracy. Needs every condition judged; ours were not (below) |
| `make_novelty_sample.py` | Builds `data/novelty_sample_ids.json` (seed 0, reproducible) |
| `notebooks/colab_ext1.ipynb` | Repeats both runs on a Colab T4 |
| `results/ext1_runs.zip` | Both runs: all 800 replies per condition, configs, logs; 669 of the control's replies are judged |
| `results/parse_only_comparison.md` | Output of `analyze_parse_only.py` |

Tests: `tests/test_ext1_diagnose_first.py` (prompts, extraction, the headline
numbers, both conditions end to end with a fake model).

## Commands

From the repository root:

```
python extension/ext1_diagnose_first/analyze_parse_only.py            # the numbers above, a few seconds
python baseline/scripts/run_all.py --config extension/ext1_diagnose_first/configs/qwen15_7b_category_first.yaml --fake --dev-subset 20
```

The real runs (GPU, about 5 to 6 s per reply on a T4, so about 1.5 hours per
condition): `notebooks/colab_ext1.ipynb`, or

```
python baseline/scripts/run_all.py --config extension/ext1_diagnose_first/configs/qwen15_7b_format_only.yaml --out runs
python baseline/scripts/run_all.py --config extension/ext1_diagnose_first/configs/qwen15_7b_category_first.yaml --out runs
```

## What full validation would need

This is a smoke test in the course's sense: one model, one seed, 800 of the
1,600 examples, prompts not piloted.

| What | Size | Cost |
|---|---|---|
| Judge the replies, for answer accuracy and refusals written in free text | about 130 control replies and about 790 diagnose-first replies | $0, about 2 days of the free Gemini quota |
| Pilot the prompts on the 800-example dev pool, then freeze them | 30 examples per prompt version, at most 3 versions | $0, under an hour of T4 time |
| Test the anchoring guess: put CLEAR last, or list the six problems before CLEAR | 2 more runs of 800 | $0, about 3 hours of T4 time |
| More seeds, and the other 800 examples | 3 seeds × 1,600 × 2 conditions | $0, about 16 hours of T4 time |

## Relation to the pitch

Diagnose first was our **first** extension idea (the "category-first" proposal).
It was run as planned, with the control, and failed. That result is the
motivation for **Extension 2** (`extension/ext2_reading_signals/`): if the
model's written diagnosis cannot be trusted, read the signal the model computes
before it writes anything.
