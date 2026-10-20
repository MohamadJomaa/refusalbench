# DSCAI team project template

Starter repository for the paper-extension project in **Data Science for
Conversational AI**. Not a lecture repository. This is where one team
reproduces a real paper's baseline, then builds and evaluates its own
novel extension of it, from day one of the course through the Module 6
defense.

## The project, in one paragraph

Pick one topic (extraction, dialogue, grounding, memory, or evaluation,
one per module) and one paper inside it, either from the course's own
curated paper-and-repo menu or proposed by your team. Reproduce that
paper's baseline. Then design, implement, and evaluate one novel,
useful extension of it, defended in Module 6 alongside two teammates who
can each answer questions about any part: the paper, the extension, or
the code. Full details, the compute and cost cap, and the rubric are in
the course syllabus and in `PROJECT-REDESIGN.md` in the course repository.

## Setup, one team, once

Do this as soon as your team has picked a topic and a paper, on whatever
connection you have, at home or wherever is reliable.

1. Click **Use this template** at the top of this repo on GitHub (not
   fork, not clone), and create your own repository, for example
   `team-alpha-gorilla-extension`. Clone it locally.
2. Add your instructor as a collaborator on your new repository, so it
   can be reviewed and graded.
3. Turn on branch protection for `main` in your new repo's settings:
   require a pull request before merging, require at least one approval.
   This is a per-repo setting and does not carry over from this template,
   so do it now, before real work starts.
4. Copy `.env.example` to `.env` and fill in whichever API key(s) your
   chosen paper's own baseline actually needs. This course's own lectures
   default to Gemini (free tier, no card required, get a key at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey)), but
   your baseline repo may expect OpenAI, Anthropic, or something else
   entirely. Check its own README before assuming. Confirm `.env` stays
   in `.gitignore` (it already is, keep it that way, never commit a real
   key).
5. `pip install -r requirements.txt` for the small set of tools every
   team needs (pytest, python-dotenv), then add whatever your chosen
   baseline repo's own `requirements.txt` or setup instructions call for
   on top of that. This template does not guess your stack for you: a
   memory paper's dependencies look nothing like an evaluation-harness
   paper's.
6. Run `python scripts/verify_setup.py`. It checks the things every
   team needs regardless of topic (`.env` loads, at least one API key is
   set, `pytest` passes) and nothing baseline-specific, since that varies
   by paper. Fix anything it reports as FAIL before your first working
   session.

## What is in here

```
baseline/
  README.md            Where your reproduction of the chosen paper's
                        baseline goes, cloned, vendored, or reimplemented
extension/
  README.md            Where your own novel idea's code goes, kept
                        clearly separate from the baseline it extends
report/
  REPORT.md            The paper-style report template, due before
                        Module 6
slides/
  README.md            What the six-content-slide deck must cover, and
                        where to build it
data/
  README.md            Where your own seed, scripted, or benchmark data
                        goes
tests/
  test_smoke.py         Confirms the expected folders exist; add real
                        tests as you build
.github/
  ISSUE_TEMPLATE/       idea-pitch, baseline-reproduction, extension, and
                        report-and-slides, one per real milestone
  PULL_REQUEST_TEMPLATE.md
CONTRIBUTING.md         The branch, PR and review workflow. Read this
                        before your first working session.
```

`baseline/README.md` and `extension/README.md` are stubs: each explains
what belongs there and what "done" looks like, no working code. That is
your team's job, not the template's.

## The four issues

Open **Issues → New issue** and work through the four templates in order:
idea pitch (approved before real work starts), baseline reproduction,
extension, then report and slides. Each already has the acceptance
criteria the syllabus sets. See `CONTRIBUTING.md` for how to assign,
split, and link work to each one.

## How this maps to the course

There is no separate practice project and then a real one. This
repository, the one you create from this template, is the same thing
from your team's first working session to the Module 6 defense. Module 6
is a live defense of whatever state this repository, your report, and
your slides are in by then, not a fresh demo built for the day.

## Data

No personal data, no client data, no real users' information. Use public
datasets, synthesized examples, or data you generate yourselves. See
`data/README.md`.
