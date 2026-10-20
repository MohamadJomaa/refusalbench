# Novelty 2 results (auto-generated): confidence-threshold refusal gate

*Generated automatically from `/content/drive/MyDrive/refusalbench-results/novelty2_confidence_threshold`. Read it together with `README.md` before writing the final `RESULT.md`.*

## Verdict (mechanical, rule fixed in the README before the run): **SIGNAL BUT NO GAIN FROM THE THRESHOLD**

- Signal check, S1 AUROC lower bound above 0.5: **yes** (AUROC 0.726 [0.699, 0.752])
- Gate check, tuned threshold beats the default in balanced accuracy with the interval above 0: **no** (difference 0.015 [-0.008, 0.036])

## Data and protocol

- 1600 examples (1064 refusal-required, 536 answerable) from 100 source questions. Model `Qwen/Qwen1.5-7B-Chat` (4bit), one forward pass per example, no generation, no judge.
- Thresholds and the probe are **cross-fitted** over source questions (5 folds, criterion `balanced_accuracy`): an example's own source question is never in the data used to tune its decision rule. Intervals are 95% cluster bootstrap over source questions (2000 resamples).
- S1 mode: `variants`. Mean probability that Qwen opens with a refusal code: 0.317. Mean probability mass on the words Yes/No in S2: 1.000 (close to 1 means Qwen answers the yes/no question as asked).

## Ranking quality (threshold-free)

| Scorer | AUROC [95% CI] | AUPRC |
|---|---|---|
| S1 refuse-code probability (the novelty) | 0.726 [0.699, 0.752] | 0.849 |
| S2 yes/no sufficiency probability | 0.751 [0.723, 0.777] | 0.857 |
| S3 hidden-state linear probe | 0.797 [0.770, 0.825] | 0.886 |

0.5 is chance, 1.0 is perfect.

## Decisions on all examples

| Decision rule | False refusal | Missed refusal | Balanced accuracy | Detection F1 |
|---|---|---|---|---|
| S1 default (p > 0.5) | 0.103 [0.065, 0.148] | 0.590 [0.548, 0.633] | 0.654 [0.631, 0.674] | 0.561 [0.521, 0.598] |
| S1 tuned threshold | 0.256 [0.197, 0.320] | 0.408 [0.358, 0.458] | 0.668 [0.642, 0.693] | 0.688 [0.653, 0.719] |
| S2 default (No > Yes) | 0.125 [0.083, 0.176] | 0.540 [0.495, 0.584] | 0.667 [0.645, 0.689] | 0.604 [0.566, 0.640] |
| S2 tuned threshold | 0.321 [0.256, 0.391] | 0.329 [0.287, 0.370] | 0.675 [0.650, 0.701] | 0.732 [0.710, 0.755] |
| S3 probe (tuned) | 0.250 [0.175, 0.327] | 0.307 [0.261, 0.356] | 0.721 [0.691, 0.749] | 0.762 [0.734, 0.786] |

Lower is better for false and missed refusal; higher is better for balanced accuracy and F1.

### Same table on the 800 novelty examples only (same predictions, subset)

| Decision rule | False refusal | Missed refusal | Balanced accuracy | Detection F1 |
|---|---|---|---|---|
| S1 default (p > 0.5) | 0.097 [0.060, 0.137] | 0.579 [0.534, 0.625] | 0.662 [0.636, 0.687] | 0.573 [0.528, 0.614] |
| S1 tuned threshold | 0.257 [0.190, 0.328] | 0.391 [0.340, 0.448] | 0.676 [0.642, 0.709] | 0.701 [0.658, 0.738] |
| S2 default (No > Yes) | 0.123 [0.078, 0.172] | 0.528 [0.478, 0.584] | 0.674 [0.646, 0.700] | 0.615 [0.567, 0.657] |
| S2 tuned threshold | 0.302 [0.234, 0.379] | 0.333 [0.285, 0.386] | 0.683 [0.646, 0.716] | 0.733 [0.697, 0.765] |
| S3 probe (tuned) | 0.220 [0.147, 0.298] | 0.308 [0.254, 0.366] | 0.736 [0.696, 0.773] | 0.767 [0.728, 0.802] |

## Differences (paired, same examples)

| Comparison | Difference [95% CI] |
|---|---|
| bal_acc: S1 tuned - S1 default | 0.015 [-0.008, 0.036] |
| f1: S1 tuned - S1 default | 0.127 [0.103, 0.153] |
| AUROC: S1 - S2 | -0.025 [-0.047, -0.003] |
| AUROC: S3 - S1 | 0.071 [0.041, 0.102] |
| bal_acc: S3 tuned - S1 tuned | 0.053 [0.024, 0.081] |

## Reference: the sampled baseline replies

On the 1560 examples present in both: clean `REFUSE_*` code on 0.431 of the refusal-required examples (missed refusal 0.569) and on 0.155 of the answerable ones (false refusal). **This is a lower bound for the baseline**: free-text refusals without a code are counted as answers because no judge was used. Treat it as context, not as the main comparison.

## By uncertainty type (S1 tuned, all examples)

| Type | refusal-required | missed refusal | answerable | false refusal |
|---|---|---|---|---|
| P-Ambiguity | 140 | 0.671 | 90 | 0.144 |
| P-Contradiction | 187 | 0.631 | 90 | 0.211 |
| P-EpistemicMismatch | 184 | 0.223 | 77 | 0.130 |
| P-FalsePremise | 185 | 0.216 | 94 | 0.543 |
| P-GranularityMismatch | 184 | 0.234 | 93 | 0.204 |
| P-MissingInfo | 184 | 0.533 | 92 | 0.272 |

## Probe configuration chosen in each fold

- fold 0: features `s1_mid`, C = 0.01
- fold 1: features `s1_mid`, C = 0.001
- fold 2: features `s2_last`, C = 0.001
- fold 3: features `s1_mid`, C = 0.01
- fold 4: features `s2_mid`, C = 0.01

## Figures

- `tradeoff.png`: false vs missed refusal for every threshold.
- `s1_distribution.png`: S1 score for the two groups.
