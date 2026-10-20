# <Your extension's name>: extending <baseline paper's short name>

**Team:** names and GitHub handles here.
**Topic:** extraction, dialogue, grounding, memory, or evaluation.
**Baseline paper:** full citation (authors, year, title, venue, arXiv ID).

## Abstract

Three to five sentences. What the baseline does, what limitation you
targeted, what you built, and the headline result of your comparison.
Write this last, once the rest of the report is finished.

## 1. The baseline paper

### 1.1 Problem

What problem does the baseline paper actually address. State it the way
the paper states it, not a generic restatement.

### 1.2 Solution

The paper's own method, at the depth needed for a reader who has not read
the paper to follow your extension. Include the method's real mechanism,
not just its name.

### 1.3 Results

The paper's own headline results, with numbers, and the benchmark or
dataset they were measured on.

## 2. Your extension

### 2.1 Motivation

The specific, named limitation of the baseline this addresses. Should
match your team's approved idea-pitch issue. If it changed since that
approval, say so and why.

### 2.2 Method

What you built, in enough detail that a reader could reproduce it from
this section alone, cross-referenced to the actual code in `extension/`.

### 2.3 Implementation notes

Anything a reader needs to run or understand your code that does not
belong in the method description itself: dependencies, data used,
configuration choices.

## 3. Results

### 3.1 Reproduction of the baseline

Your team's own reproduced number(s) next to the paper's original
number(s). If a smoke test stood in for a full run, say so here and state
what a full-scale run would require.

### 3.2 Extension versus baseline

A direct, real comparison: your extension's results against the baseline
you reproduced, on the same measure. A table is usually the clearest way
to show this.

### 3.3 Honest assessment

What worked, what did not, and why, in your own judgment. This section
exists specifically so a team does not present only the numbers that look
good.

## 4. Related work

Cite any published follow-up work on the same baseline paper you found
while researching your idea (the idea-pitch issue's own related-work
check). State explicitly how your extension differs from each one you
cite. If you found none, say so plainly rather than leaving this section
thin without comment.

## 5. Limitations and what full validation would require

If any part of this project used a smoke test in place of full-scale
validation (per the project's compute and cost cap), state exactly which
part, and what running it at full scale would actually require: data
volume, compute, time, or cost. This is not a penalty as long as it is
stated honestly.

## References

Full citations for the baseline paper, every related-work paper cited in
Section 4, and any other source used.
