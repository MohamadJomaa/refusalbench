# Baseline

Your reproduction of the chosen paper's own method goes here. Three
honest ways to fill this folder, in order of preference:

1. **Clone or vendor the paper's own repo as a subfolder**, if one
   exists (check `research/PROJECT-PAPER-MENU.md` in the course repo for
   verified links, or the paper itself if self-proposed). Keep its own
   structure and README intact; add a short note at the top of this file
   pointing to it and stating any changes your team made to get it
   running.
2. **Reimplement the method yourselves**, if the repo is unusable
   (archived, broken, missing a critical piece) or your paper has no
   repo at all. Say so explicitly here, this is real extra work the
   idea-pitch issue should already have flagged.
3. **Wrap a hosted or pip-installable version** of the method (a paper
   whose method shipped as a maintained package, Mem0 or Zep/Graphiti for
   example), pointing your own code at it rather than vendoring source.

Whichever path, this folder's own README (replace this file, or add
alongside it) must state:

- Exactly which commands reproduce which of the paper's own results, or
  a named subset of them if a smoke test stood in for the full run.
- Your team's own reproduced number(s) next to the paper's original
  number(s), so a reader can see the comparison without re-running
  anything.
- If a smoke test was used, what a full-scale run would actually
  require (data volume, compute, time), stated plainly, not skipped.

Nothing in this folder should be your own novel idea. That belongs in
`extension/`, kept clearly separate so a reader (or a grader) can tell at
a glance what is the reproduced paper and what is your team's own work.
