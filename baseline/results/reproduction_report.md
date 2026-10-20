# Reproduction report: Qwen1.5-7B-Chat on RefusalBench-NQ

*Generated automatically. The assessment below is provisional until we review it together.*

*Regenerated on 20 October 2026 from the same graded replies with the current code, which maps the judge's two mistyped refusal codes to the closest valid code; only detection F1, recall and the refusal rate on unanswerable change. The environment line is the original run's.*

> **INCOMPLETE JUDGING: 790 of 1560 replies are not judged yet. Do not interpret or compare these numbers.**


## Objective
Reproduce, as closely as practical on free resources, the paper's evaluation of Qwen1.5-7B-Chat on the
released RefusalBench-NQ benchmark (1,600 examples), and compare the measured selective-refusal metrics.

## Original setup (paper)
Qwen/Qwen1.5-7B-Chat served with vLLM on A100 GPUs, temperature 1.0, top_p 1.0. The target model replies with an
answer or a `REFUSE_*` code. Judge: Claude-4-Sonnet classifies each reply (answer attempt or refusal code) and scores
answer quality 1-5; a score of 4 or 5 is a correct answer, a refusal is correct on an exact category match.

## Our setup
- Target: `Qwen/Qwen1.5-7B-Chat`, precision `4bit`, temperature 1.0, top_p 1.0, max_new_tokens 256, seed 0, batch size 8
- Prompt: copied verbatim from the original repo (hash `c0008c7504b3`), sent as a single user message
- Judge: `gemini-3.5-flash-lite` (gemini), temperature 0.0. Replies with one clean `REFUSE_*` code are scored by code; every other reply goes to the judge
- Environment: {"python": "3.13.15", "platform": "Linux-6.6.122+-x86_64-with-glibc2.39", "torch": "2.11.0+cpu", "transformers": "5.17.0", "bitsandbytes": null, "accelerate": "1.15.0", "google-genai": "2.12.1", "datasets": "4.8.5", "gpu": null}

## Deviations from the paper
1. Quantization: `4bit` on a Colab T4, the paper ran vLLM on A100 GPUs (likely full precision)
2. Inference stack: Hugging Face transformers instead of vLLM v0.5.1; the chat template adds Qwen's default system prompt
3. Judge: `gemini-3.5-flash-lite` instead of Claude-4-Sonnet, and only unclear replies are judged (clean refusal codes are scored by code)
4. max_new_tokens 256 (not stated in the paper); sampling at temperature 1.0 means results vary run to run
5. Dataset: the released Hugging Face version of RefusalBench-NQ

## Data integrity
Generated 1560 replies; judged/scored 770; unjudged 790; empty replies 0.
Reply format: {'clean_code': 528, 'no_code': 991, 'unknown_code': 40, 'multiple_codes': 1} (clean `REFUSE_*` code rate 33.8%).
Judge calls: 498, scored by parser: 272.


## Missing examples (excluded)
40 of 1600 examples (2.5%) have no reply because the GPU ran out of
memory on their batch and they were never regenerated. They are excluded from every metric, not counted as errors.
- By expected behavior: {'REFUSE_INFO_MISSING_IN_CONTEXT': 2, 'REFUSE_NONFACTUAL_QUERY': 7, 'REFUSE_FALSE_PREMISE_IN_QUERY': 5, 'REFUSE_GRANULARITY_MISMATCH': 5, 'ANSWER_CORRECTLY': 13, 'REFUSE_AMBIGUOUS_QUERY': 1, 'REFUSE_CONTRADICTORY_CONTEXT': 7}
- By uncertainty type: {'P-MissingInfo': 3, 'P-EpistemicMismatch': 8, 'P-FalsePremise': 10, 'P-GranularityMismatch': 7, 'P-Ambiguity': 4, 'P-Contradiction': 8}
- Median length (question plus context, characters): missing 670 vs scored 372.
  The missing examples are somewhat longer, so the sample is slightly biased toward shorter inputs.

## Results (overall)
| metric | value |
|---|---:|
| answerable / unanswerable examples | 261 / 509 |
| answer accuracy | 0.690 |
| refusal accuracy (exact category) | 0.114 |
| false refusal rate | 0.165 |
| missed refusal rate | 0.456 |
| refusal rate on unanswerable (any code) | 0.544 |
| refusal detection F1 (refuse = positive) | 0.668 |
| calibrated refusal score (0.5 x answer + 0.5 x refusal accuracy) | 0.402 |

### By uncertainty type
| group | n | answer acc | refusal acc | false refusal | missed refusal | detection F1 |
|---|---:|---:|---:|---:|---:|---:|
| P-Ambiguity | 120 | 0.745 | 0.027 | 0.085 | 0.740 | 0.396 |
| P-Contradiction | 132 | 0.690 | 0.044 | 0.167 | 0.711 | 0.423 |
| P-EpistemicMismatch | 131 | 0.721 | 0.080 | 0.116 | 0.295 | 0.800 |
| P-FalsePremise | 128 | 0.550 | 0.057 | 0.325 | 0.193 | 0.826 |
| P-GranularityMismatch | 125 | 0.750 | 0.000 | 0.104 | 0.247 | 0.829 |
| P-MissingInfo | 134 | 0.659 | 0.430 | 0.220 | 0.559 | 0.573 |

### By intensity
| group | n | answer acc | refusal acc | false refusal | missed refusal | detection F1 |
|---|---:|---:|---:|---:|---:|---:|
| HIGH | 253 | n/a | 0.126 | n/a | 0.395 | 0.754 |
| LOW | 261 | 0.690 | n/a | 0.165 | n/a | n/a |
| MEDIUM | 256 | n/a | 0.102 | n/a | 0.516 | 0.653 |

Confusion matrix: `confusion_matrix.csv`.

## Comparison with the paper
| metric | paper (Qwen1.5-7B-Chat) | ours | difference | within +/-0.05 |
|---|---:|---:|---:|---|
| answer_accuracy | 0.561 (exact, Section 4.2 text, effect of scale) | 0.690 | +0.129 | no |
| refusal_accuracy | 0.05 (approximate, read from the small Figure 9 plot; text says < 0.17 for all Qwen sizes) | 0.114 | +0.064 | no |
| false_refusal_rate | not extracted from the paper | 0.165 | n/a | n/a |
| missed_refusal_rate | not extracted from the paper | 0.456 | n/a | n/a |
| calibrated_refusal_score | not extracted from the paper | 0.402 | n/a | n/a |
| refusal_detection_f1 | not extracted from the paper | 0.668 | n/a | n/a |

Metrics the paper reports only in charts we did not extract are marked, not guessed.

## Reproducibility assessment (provisional)
**INCOMPLETE: 790 of 1560 replies are not judged yet, so these numbers cover only part of the data and must not be interpreted or compared with the paper** (rule: a metric counts as reproduced when within +/-0.05 of the paper value; the paper's refusal
accuracy is itself an approximate read of a small chart). Not a verdict until we review the results and the
judge hand check (`handcheck_sheet.csv`, 30 items to fill in) together.

## Observed failure patterns (descriptive only)
Most common wrong refusal categories:
- expected `REFUSE_FALSE_PREMISE_IN_QUERY` but got `REFUSE_INFO_MISSING_IN_CONTEXT`: 65 times
- expected `REFUSE_CONTRADICTORY_CONTEXT` but got `answer_attempt`: 64 times
- expected `REFUSE_GRANULARITY_MISMATCH` but got `REFUSE_INFO_MISSING_IN_CONTEXT`: 57 times
- expected `REFUSE_AMBIGUOUS_QUERY` but got `answer_attempt`: 54 times
- expected `REFUSE_INFO_MISSING_IN_CONTEXT` but got `answer_attempt`: 52 times
