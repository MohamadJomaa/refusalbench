# Idea pitch: reading a small model's refusal signal instead of its text

**Team project, Grounding topic. Paper: RefusalBench (EACL 2026). Model: Qwen1.5-7B-Chat on a free Colab T4.**

## The pitch (one paragraph)

RefusalBench tests whether a grounded language model knows when to refuse because the evidence is ambiguous, contradictory, missing, based on a false premise, at the wrong granularity, or asks for an opinion. We reproduced the baseline on the released RefusalBench-NQ set (1,600 examples) with Qwen1.5-7B-Chat in 4-bit: the model misses a large share of the cases where it should refuse and almost never names the right reason, as the paper also reports for small Qwen models. Our first extension, asking Qwen to write an evidence-state diagnosis before answering, failed clearly: Qwen wrote "CLEAR" on 95% of the examples and refused almost never, so it was worse than the baseline (we keep this as a documented negative result, with a format-only control). Our proposed extension reads the model instead of trusting its text: with a single forward pass and no generation, we take the probability that Qwen opens a refusal code (and, as ablations, a yes/no sufficiency probability and a linear probe on its last hidden state) and turn it into an answer-or-refuse decision with a threshold. We evaluate it on the same benchmark against the baseline with a paired bootstrap over the 100 source questions, thresholds and probes cross-fitted so that no source question is in its own training data, and we report false refusal, missed refusal, balanced accuracy and detection F1. In a pilot run the refusal probability separates the two groups clearly above chance (AUROC about 0.73, probe about 0.80), but a simple threshold alone does not beat the model's default behavior, and a first comparison with the judged baseline shows the probe only slightly ahead (about 3 points of balanced accuracy, not yet statistically clear), so the open question we propose to answer is whether a hidden-state probe makes better answer-or-refuse decisions than the baseline once the baseline is fully judged. We present it as a lightweight proof of concept for one small model, not a general solution.

## What we ask you to approve

1. Novelty 2 (reading the refusal probability and a hidden-state probe) as our extension, with the failed prompt-based attempt reported as motivation.
2. That we frame the claim narrowly: better *detection* of when to refuse (answer vs refuse), not better *explanation* of why.

## Related work to cite and position against (verify each before submission)

- Probing hidden states for what a model "knows" or whether it is uncertain, for example Kadavath et al., 2022 (*Language Models (Mostly) Know What They Know*), Burns et al., 2022 (*Discovering Latent Knowledge*) and Azaria and Mitchell, 2023 (*The Internal State of an LLM Knows When It's Lying*).
- Selective prediction and abstention with confidence thresholds.
- The RefusalBench paper itself, and its analysis that detection and categorization are separate abilities.

How we differ: we apply the probe and threshold idea to grounded selective refusal on RefusalBench with a small open model, evaluate it against the paper's baseline with a source-question cluster bootstrap, and report the prompting variant that did not work.

## Honest status

- Baseline: reproduced for Qwen1.5-7B-Chat; replies generated for 1,560 of 1,600 examples; 770 of them graded so far (random sample), the rest in progress (free Gemini quota).
- Novelty 1: done, negative result.
- Novelty 2: pilot run done on all 1,600 examples. First fair comparison against the judged baseline (770 graded replies): the probe is +0.028 [-0.014, +0.065] in balanced accuracy, no clear difference yet; the label-free threshold rules do not beat the baseline. The comparison will be repeated when grading is complete. We do not claim an improvement before that.
