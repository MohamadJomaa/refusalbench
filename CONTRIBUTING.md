# How this team works together

Three rules. They exist to solve three different problems, so do not skip one because another seems to cover it.

## 1. One branch per person, one pull request per person

Problem this solves: two people editing the same file at the same time.

Each person works on their own branch, named `<your-name>/<what-you-are-doing>`, for example `alice/baseline-eval-loop` or `bilal/extension-ablation`. When your piece is ready, open a pull request into `main` yourself. Do not commit directly to `main`.

If two people are working on the same issue, split it by file or by function before you start, not after you both have changes. Say the split out loud in the issue thread so it is written down.

## 2. Every pull request needs one teammate's approval before merging

Problem this solves: knowing who actually wrote what, for both learning and grading.

Review each other's pull requests for real: read the diff, run it, ask a question if something is unclear. A rubber-stamp approval defeats the point. Your instructor can see every PR's author and every review, and will use that history during the Module 6 defense to ask each of you specifically about a part of the project a teammate built, not the part you built.

## 3. Always commit under your own GitHub account

Problem this solves: making the above two rules mean anything.

Never share a login. Never commit on someone else's behalf, even to help them out. If you paired on something, both of you push your own commits from your own branch, or note it explicitly in the PR description.

## The four issues

Open **Issues → New issue** and work through the four templates in order: idea pitch, baseline reproduction, extension, then report and slides. Before you start work on an issue:

1. Assign it to every teammate who will touch it.
2. Fill in the per-person checklist inside the issue with who is doing which part.
3. Link your pull requests to the issue (`Closes #<number>` in the PR description) so the issue closes itself when the last one merges.

The idea-pitch issue in particular should not be opened as a formality: it is the one checkpoint that catches an extension idea that already exists as someone else's published follow-up paper, before your team has spent real time on it. Get the instructor's approval on it before treating the baseline-reproduction or extension work as started.

## Before Module 6

Re-read the syllabus rubric. Any of the three of you can be asked about any part of the project at the defense, the baseline paper, the extension, or the code, not only the part you personally wrote. Shared ownership is the point.
