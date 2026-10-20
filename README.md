# Reading, Not Asking: selective refusal in a small grounded model

Team project for **Data Science for Conversational AI** (MSc, Lebanese University).
Topic: **Grounding**. Baseline paper: **RefusalBench** (Muhamed et al., EACL 2026).

**Team:** Mohammad Jomaa (`@<github-handle>`)

## The project, in one paragraph

RefusalBench tests whether a grounded language model answers when the given
passage supports an answer, and refuses, naming the right reason, when the
passage is flawed (six reasons: ambiguity, contradiction, missing information,
false premise, granularity mismatch, epistemic mismatch). We **reproduce** the
paper's evaluation for a small open model, **Qwen1.5-7B-Chat**, on the released
RefusalBench-NQ set (1,600 examples) on a free Google Colab T4, with a Gemini
judge in place of the paper's Claude judge. We then test **two extensions**:

1. **Diagnose first** (`extension/ext1_diagnose_first/`): ask the model to state
   what is wrong with the evidence before it answers. A clear negative result:
   the model wrote "CLEAR" on 762 of 800 examples.
2. **Read the model's signal** (`extension/ext2_reading_signals/`): with one
   forward pass and no generated text, read the probability of a refusal code,
   a yes/no sufficiency probability, and a linear probe on the hidden state,
   and turn each into an answer-or-refuse decision. The signal is clearly above
   chance (AUROC 0.73 to 0.80), but a better decision than the baseline is
   **not** established (probe +0.028 balanced accuracy, 95% interval
   [-0.014, +0.065]).

## Where everything is

```
baseline/     Our reproduction of the RefusalBench evaluation (paper's prompt and
              judge prompt verbatim), configs, Colab notebook, results
extension/    Our own work, kept apart from the baseline:
  ext1_diagnose_first/    Extension 1: evidence-state prompt + format-only control
  ext2_reading_signals/   Extension 2: refusal probability, yes/no, hidden-state probe
data/         Where the data comes from, and the fixed sample ID files
report/       REPORT.md, the paper-style report
slides/       The defense deck (cover, 6 content slides, references)
tests/        pytest tests: run without a GPU, an API key or a network
scripts/      verify_setup.py, the setup check
```

## Status

| Part | State |
|---|---|
| Baseline | 1,560 of 1,600 replies generated (40 out-of-memory failures, 2.5%); **770 graded** (a seeded random sample), the rest limited by the free judge quota |
| Extension 1 | Done, negative result, on a stratified 800 |
| Extension 2 | Done on all 1,600; compared with the baseline on the 770 graded examples |
| Report and slides | See `report/` and `slides/` |

Every result is a **smoke test** in the course's sense: what a full run would
need is stated in `baseline/README.md`, `extension/README.md` and Section 5 of
the report.

## Setup

1. Clone the repository and create a Python 3.10+ environment.
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY` (free key from
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey)). It is
   needed only for judging; generation and the tests do not use it.
4. `python scripts/verify_setup.py` and fix every FAIL.

**Where it runs.** The tests and the `--fake` end-to-end runs work on any
laptop CPU. The real runs need a GPU with about 6 GB of free memory for
Qwen1.5-7B-Chat in 4-bit: we used the free Google Colab T4 through the
notebooks in `baseline/` and `extension/`. Total API cost: **$0** (free Gemini
tier), inside the course's $20 limit.

## Documented deviations from the paper

| | Paper | Ours |
|---|---|---|
| Hardware | 4 × A100 80 GB, vLLM | 1 × Colab T4, Hugging Face transformers |
| Precision | not stated (likely 16-bit) | 4-bit (bitsandbytes) |
| Judge | Claude-4-Sonnet | `gemini-3.5-flash-lite`, used only for replies without one clean refusal code |
| Sampling | temperature 1.0, top_p 1.0 | same |
| Prompt and judge prompt | original repository | copied verbatim |

## Team workflow

See `CONTRIBUTING.md`: one branch and one pull request per person, every pull
request approved by a teammate, every commit under its author's own account.
The four milestones are the four issue templates under **Issues → New issue**.

## Acknowledgements and links

- Paper: Muhamed et al., *RefusalBench: Generative Evaluation of Selective
  Refusal in Grounded Language Models*, EACL 2026, arXiv:2510.10390.
- Original code: [github.com/aashiqmuhamed/refusalbench](https://github.com/aashiqmuhamed/refusalbench)
- Our earlier working repository (history of the runs):
  [github.com/aidepartmentrah-pixel/refusalbench-reproduction](https://github.com/aidepartmentrah-pixel/refusalbench-reproduction)
- Course template: [github.com/ahmabboud/dsca-team-template](https://github.com/ahmabboud/dsca-team-template)

## Data

No personal data. RefusalBench-NQ is built from Natural Questions and KILT
(Apache 2.0). See `data/README.md`.
