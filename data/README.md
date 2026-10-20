# Data

We use one public benchmark and generate no data of our own. This folder holds
only the two fixed lists of example IDs that make our samples reproducible.

## The benchmark: RefusalBench-NQ

| | |
|---|---|
| Source | Hugging Face dataset [`aashiqmuhamed/RefusalBench-NQ`](https://huggingface.co/datasets/aashiqmuhamed/RefusalBench-NQ), split `test` |
| Paper | Muhamed et al., *RefusalBench*, EACL 2026, arXiv:2510.10390 |
| Built from | 100 questions from Natural Questions (Kwiatkowski et al., 2019), with ground-truth passages from KILT (Petroni et al., 2020). The paper states both are Apache 2.0; check the dataset card for the release's own license |
| Size | **1,600 examples**: **536 answerable** (all LOW intensity) and **1,064 that must be refused** (MEDIUM and HIGH) |
| Uncertainty types | ambiguity, contradiction, missing information, false premise, granularity mismatch, epistemic mismatch |
| Personal data | none: Wikipedia passages and generated variants of them. The authors also checked hundreds of examples by hand for personal information and found none |

**The dataset is not stored in this repository.** `baseline/refusalbench_repro/dataset_loader.py`
downloads it on every run (`datasets.load_dataset`, cached locally by the
`datasets` library), and `baseline/scripts/run_all.py` stops if it does not get
exactly 1,600 rows with 536 answerable. Nothing is filtered or changed.
`schema_adapter.py` only renames the released fields:

| Released field | Our field |
|---|---|
| `id`, `source_id` | `id`, `source_id` (the source question, used to keep variants of one question together) |
| `perturbed_query`, `perturbed_context` | `question`, `context` |
| `expected_rag_behavior` | `expected_behavior` (`ANSWER_CORRECTLY` or one `REFUSE_*` code) |
| `perturbation_class`, `intensity` | `uncertainty_type`, `intensity` |
| `original_answers` | `reference_answers` (answerable rows only) |
| `generator_model` | `generator_model` |

All 1,600 examples are variants of only **100 source questions**, so they are not
independent. That is why our samples keep every source question, and why our
confidence intervals resample whole source questions.

## The files in this folder

Both are JSON, seeded (seed 0), and list example IDs only, no text.

### `smoke_sample_ids.json`: 20 examples

A small stratified sample for dry runs. Used by `baseline/scripts/run_smoke_test.py`
(and created by it if missing). 7 answerable and 13 to be refused; all six
uncertainty types; LOW 7, MEDIUM 6, HIGH 7; 18 source questions.

### `novelty_sample_ids.json`: 800 examples, plus an 800-example dev pool

The fixed sample for **Extension 1**: the format-only control and the
diagnose-first condition both run on exactly these `ids`, and the baseline is
compared on the same examples. Extension 2 scores all 1,600 and reports this
800 as a subset.

- `ids`: **800** examples. Half of every stratum (answerable × type, and
  refusal code × intensity), rounding settled by a seeded draw.
  **268 answerable, 532 to be refused**; LOW 268, MEDIUM 265, HIGH 267;
  **all 100 source questions** (2 to 15 examples each); by generator: Claude
  194, DeepSeek 190, GPT 200, Nova 216.
- `dev_pool_ids`: the **other 800**, no overlap with `ids`, also covering all
  100 source questions. Reserved for prompt pilots. **Not used in any reported
  result.**

Built by `extension/ext1_diagnose_first/make_novelty_sample.py`. Running it
again with seed 0 gives the same file.

## Where the run outputs are

Model replies and judge grades are not in this folder. They are in
`baseline/results/` and `extension/*/results/`, as zips. They contain copies of
the benchmark's questions and passages next to the model's replies, and no
other data.

## Rules

- No personal data, no client data, no real users' information.
- Everything in this folder except this README is gitignored by the template.
  The two ID files were added on purpose with `git add -f`. Do the same for any
  new file, and keep it small.
