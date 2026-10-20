# Extension 2: read the model's signal instead of its written answer

**Verdict: the signal exists; a better decision than the baseline is not
established.** With one forward pass and no generated text, three readouts of
Qwen1.5-7B-Chat separate the examples that need a refusal from the answerable
ones far above chance (AUROC 0.73 to 0.80). But turned into answer-or-refuse
decisions, none clearly beats the baseline. The best, a linear probe on the
hidden state, is **+0.028** balanced accuracy, 95% interval **[-0.014, +0.065]**,
which includes zero. **We claim no improvement.**

## The limitation it targets, and why this idea

The limitation is the one in `../README.md`: a small grounded model misses many
refusals (our baseline: 45.6% of the needed refusals are answered instead) and
the paper offers no fix without training.

Extension 1 asked the model to write down what was wrong with the evidence,
and the model wrote "CLEAR" 95% of the time. So this extension **does not ask:
it reads.** If the model's written answer cannot be trusted, look at what it
computes before it writes. The question is narrower than the paper's: **when**
to refuse (detection), not **why** (the category), the skill the paper shows
small models are weakest at.

## What we built

Every readout gives one number per example; higher always means "refuse".

| Readout | What is read | Labels used? |
|---|---|---|
| **S1** refusal probability | Under the paper's own prompt, the probability that the reply opens with a refusal code. A code is several tokens, and Qwen sometimes wraps it (`**REFUSE`, a backtick, `ANSWER: REFUSE`), so for each wrapper we take the longest token prefix shared by all seven codes, score it by teacher forcing, and sum over wrappers | no, at the default threshold (p > 0.5) |
| **S2** yes/no | A second prompt: "Can the query be answered completely and faithfully from the context?" Log-odds of `No` against `Yes` | no, at the default threshold (No > Yes) |
| **S3** linear probe | Logistic regression (standardise, PCA to 128 components, class-balanced) on Qwen's last-token hidden state from either prompt, last or middle layer, chosen by inner grouped cross-validation | **yes**: trained on the benchmark's labels |

Each readout becomes a decision with a threshold. Two kinds of rule:
**default thresholds** (S1, S2) use no labels at all, the cleanest comparison
with the baseline; **tuned thresholds** and the probe are fitted on labels.

**No leakage.** The 1,600 examples are variants of 100 source questions.
Everything fitted (thresholds, the probe, its settings) is **cross-fitted over
source questions in 5 groups**: an example's own source question is never in
the data used for its threshold or its probe. A unit test corrupts the labels
of one group and checks that its threshold does not move.

**The success rule was fixed before the run** (`analysis.py`, `verdict`):
1. **Signal:** the lower end of S1's AUROC interval is above 0.5.
2. **Gain:** the tuned S1 threshold beats the default one in balanced
   accuracy, with the interval of the difference above zero.

Both: SUPPORTED. Only the first: SIGNAL BUT NO GAIN. Neither: NOT SUPPORTED.

Code: `scorer.py` (the forward passes, the token probabilities, hidden states,
resumable shards), `analysis.py` (cross-fitting, thresholds, probe, bootstrap,
report, figures), `run_ext2.py` (the one entry point), `compare_to_baseline.py`
(the comparison with the judged baseline). Settings: `configs/ext2.yaml`.

## How it is evaluated against the baseline

On **the same 770 examples** the baseline has graded replies for (509 must
refuse, 261 answerable, 99 source questions). The baseline's decision is read
from the judge: any `REFUSE_*` classification is a refusal (a free-text
refusal the judge recognised counts), `answer_attempt` and `EMPTY` answer.

Measure: **balanced accuracy** = 1 − (false refusal rate + missed refusal
rate) / 2, so 0.5 is a coin flip whatever the class balance (two thirds of the
examples must be refused). Intervals: cluster bootstrap over source questions,
2,000 resamples. **BETTER** when the whole interval of (rule − baseline) is
above 0, **WORSE** when below, otherwise **NO CLEAR DIFFERENCE**.

## Results

### A. Is the information inside the model? (all 1,600 examples)

| Readout | AUROC [95% interval] |
|---|---|
| S1 refusal probability | 0.726 [0.699, 0.752] |
| S2 yes/no | 0.751 [0.723, 0.777] |
| S3 probe | 0.797 [0.770, 0.825] |

0.5 is chance. All three are clearly above it. **Pre-registered verdict:
SIGNAL BUT NO GAIN** (signal check passed, lower end 0.699; the tuned S1
threshold beat the default by +0.015 [-0.008, +0.036], which includes zero).

### B. Does using it beat the baseline? (770 graded examples)

From `results/comparison_all/COMPARISON.md`:

| Rule | False refusal | Missed refusal | Balanced accuracy | Minus baseline [95%] | Verdict | Labels? |
|---|---:|---:|---:|---:|---|---|
| **Baseline** (judged replies) | 0.165 | 0.456 | **0.690** | | | no |
| S1 default | 0.103 | 0.580 | 0.658 | -0.031 [-0.057, -0.005] | **WORSE** | no |
| S1 tuned | 0.261 | 0.397 | 0.671 | -0.018 [-0.047, +0.009] | no clear difference | yes |
| S2 default | 0.115 | 0.546 | 0.669 | -0.020 [-0.047, +0.007] | no clear difference | no |
| S2 tuned | 0.337 | 0.326 | 0.668 | -0.021 [-0.056, +0.014] | no clear difference | yes |
| **S3 probe** | 0.276 | **0.289** | **0.718** | **+0.028 [-0.014, +0.065]** | no clear difference | yes |

With five comparisons and no correction for multiple testing, even S1's
"worse" is not robust.

**Why a good AUROC does not give a better decision.** AUROC measures ranking
over every possible threshold; a deployed rule needs one operating point, and
moving the threshold only trades one error for the other. In the trade-off
figure (`results/comparison_all/baseline_vs_scorers.png`) the baseline sits on
the S1 and S2 curves and just above the probe's.

**By type** (missed / false refusal, same 770): the probe finds many more
contradictions (0.71 / 0.17 for the baseline, 0.19 / 0.29 for the probe) and
ambiguities (0.74 / 0.09, then 0.45 / 0.23), the two types the baseline misses
most, but refuses more answerable questions. False premise goes the other way
(0.19 / 0.33, then 0.25 / 0.23). Full table in `COMPARISON.md`.

## Files

| File | What it is |
|---|---|
| `scorer.py`, `analysis.py`, `run_ext2.py`, `compare_to_baseline.py` | The code (above) |
| `configs/ext2.yaml` | All settings: 5 folds, seed 0, 2,000 resamples, threshold criterion, probe grid. `run_name` stays `novelty2_confidence_threshold`, the name of the run folder on Drive, so that run can still resume |
| `notebooks/colab_ext2.ipynb` | The whole run on a Colab T4 (no judge, no API key) |
| `PITCH.md` | The idea pitch for this extension, written for the instructor |
| `results/scores.csv` | Every example: its source question, label, fold, the three scores (the probe's out-of-fold), every rule's decision |
| `results/RESULT_AUTO.md`, `metrics.json`, `tradeoff.png`, `s1_distribution.png`, `run.log` | The run's own report (part A, the mechanical verdict), numbers, figures, log |
| `results/comparison_all/` | Part B: `COMPARISON.md`, `comparison.json`, `baseline_vs_scorers.png` |
| `results/ext2_run.zip` | The raw download of the run from Colab |

Tests: `tests/test_ext2_analysis.py` (metrics by hand, folds, no leakage, the
verdict on synthetic data with a known answer), `tests/test_ext2_compare.py`
(the comparison rule, and parts A and B **recomputed from the committed
files**), `tests/test_ext2_run.py` (the whole run with a fake scorer, and the
scorer on a tiny random Qwen2 when torch is installed).

## Commands

From the repository root:

```
python extension/ext2_reading_signals/compare_to_baseline.py --out runs/ext2_comparison   # part B, seconds, no GPU
python extension/ext2_reading_signals/run_ext2.py --fake --limit 60 --out runs             # the pipeline, no GPU
```

The real run (GPU; about **0.3 s per example** per prompt on a T4, so about
15 minutes of scoring plus the model download): `notebooks/colab_ext2.ipynb`, or

```
python extension/ext2_reading_signals/run_ext2.py --out runs
```

The committed `results/` cannot be re-analysed from scratch without
re-scoring: the hidden states (`shards/`, needed to refit the probe) stayed on
Drive and are not in the repository. `scores.csv` keeps everything the
reported numbers need.

## What full validation would need

A smoke test in the course's sense: one model, one seed, a half-graded
baseline.

| What | Size | Cost |
|---|---|---|
| Grade the other **790** baseline replies, then rerun `compare_to_baseline.py` | about 534 judge calls | $0, one more day of the free quota. With all 1,560 the intervals narrow, so the probe's +0.028 may become a small clear gain or stay unclear |
| Check the Gemini judge by hand | 30 replies (`baseline/results/handcheck_sheet.csv`) | an hour of reading |
| An equal-training comparison | give the baseline the same labels: sample 5 replies per example and tune a threshold on the share that refuse, cross-fitted the same way | $0, about 13 hours of T4 time for all 1,600 (6.5 hours for the 800) |
| Correct for the five comparisons | Holm or Bonferroni on the five intervals | $0, analysis only |
| Other fold splits | rerun the analysis with fold seeds 1 to 4 (the scores themselves are deterministic: no sampling) | $0, minutes, but needs the hidden-state shards from Drive |
| Other models, and the multi-document RefusalBench-GaRAGe | 2 more models × 1,600 examples | $0 on Colab, under an hour of T4 time per 7B model |

## Relation to the pitch

This extension replaced Extension 1 after Extension 1 failed. `PITCH.md` is
the pitch that asks the instructor to approve the switch, and to accept the
narrow claim: better **detection** of when to refuse, not better explanation
of why.

## Related work

Reading what a model "knows" from its probabilities or its hidden states is
established: Kadavath et al. (2022) on self-evaluation from probabilities,
Burns et al. (2023) and Azaria and Mitchell (2023) on probing hidden states,
Alain and Bengio (2016) on linear probes; selective prediction with a
confidence threshold goes back to Geifman and El-Yaniv (2017). Methods that fix
refusal by **training** the model include R-Tuning (Zhang et al., 2024) and
Self-RAG (Asai et al., 2023). **How we differ:** we apply reading-not-asking to
grounded selective refusal on RefusalBench, with a small open model whose
weights stay fixed, compare it against the paper's own baseline with intervals
that respect the 100 source questions, and report the prompting version that
did not work (Extension 1). The full discussion is in `report/REPORT.md`,
Section 4.
