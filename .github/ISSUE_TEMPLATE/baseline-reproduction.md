---
name: "2: Baseline reproduction"
about: Reproduce the chosen paper's own baseline before building your extension on top of it
title: "Baseline reproduction: <your paper>"
labels: baseline
---

## Objective

Get the baseline paper's own method actually running, in `baseline/`, and
reproduce at least one of its own headline results closely enough to
trust as a starting point for your extension. This is not a from-scratch
reimplementation unless your chosen paper genuinely has no usable repo,
in that case, say so in the idea-pitch issue and budget real extra time
here.

## Acceptance criteria

- [ ] The baseline runs end to end on your own machine or account, documented in `baseline/README.md` with the exact commands
- [ ] At least one number from the paper's own results is reproduced, or a smoke-test subset of it, with your team's own number placed next to the paper's for comparison
- [ ] If a smoke test stood in for full-scale validation, `baseline/README.md` states plainly what full-scale validation would require (data volume, compute, time), not glossed over
- [ ] Confirmed within the compute and cost cap: runs on a personal computer, or the API spend so far is tracked and under $20

## Who is doing what

Replace this with real names before starting work. Every checklist item needs exactly one owner.

- [ ] @person-1:
- [ ] @person-2:
- [ ] @person-3:

## Pull requests closing this issue

Link them here as they open, for example: `Closes #<number>` in the PR description does this automatically.
